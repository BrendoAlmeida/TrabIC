"""Resolve alocação de candidatos por escola e sala com Gurobi (WLS)."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.instancias import ler_instancia
from src.otimizacao.busca_multiexchange import busca_multiexchange
from src.otimizacao.busca_tabu import busca_tabu
from src.otimizacao.construtivo import construir
from src.otimizacao.salas import SolucaoSalas, avaliar_salas, verificar_salas


def _gurobi():
    try:
        import gurobipy as gp
    except ImportError as erro:
        raise RuntimeError(
            "Gurobi não está instalado neste ambiente. Rode "
            "'.venv/bin/python -m pip install -r requirements-gurobi.txt'."
        ) from erro
    return gp


def verificar_licenca() -> str:
    """Inicializa um ambiente Gurobi sem exibir nem ler credenciais próprias."""
    gp = _gurobi()
    with gp.Env():
        pass
    versao = ".".join(str(x) for x in gp.gurobi.version())
    return versao


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _ler_solucao_json(path: Path, instancia, k: int) -> SolucaoSalas:
    dados = json.loads(path.read_text(encoding="utf-8"))
    if dados.get("k") != k:
        raise ValueError(f"A solução em {path} usa K diferente de {k}.")
    pares = dados.get("solucao")
    if pares is None:
        pares = dados.get("alocacao")
    if not isinstance(pares, list) or any(
            not isinstance(p, (list, tuple)) or len(p) != 2 for p in pares):
        raise ValueError(f"{path} não contém uma alocação (escola, sala) por candidato.")
    solucao = SolucaoSalas(tuple(tuple(p) for p in pares))
    verificar_salas(instancia, solucao, k)
    custo = avaliar_salas(instancia, solucao)
    custo_arquivo = dados.get("custo")
    if custo_arquivo is None:
        custo_arquivo = dados.get("custo_final")
    if custo_arquivo is not None and not math.isclose(
            custo, float(custo_arquivo), rel_tol=1e-8, abs_tol=1e-6):
        raise ValueError(f"O custo da solução em {path} não confere com a instância.")
    return solucao


def _melhor_start_salvo(instancia, k: int, instance_path: Path):
    diretorio = ROOT / "resultados"
    if not diretorio.is_dir():
        return None
    padrao = f"run_{instance_path.name}_{k}_multistart_tabu_multiexchange_*.json"
    hash_instancia = _sha256(instance_path)
    melhores = []
    for path in diretorio.rglob(padrao):
        try:
            dados = json.loads(path.read_text(encoding="utf-8"))
            if (dados.get("instance") != instance_path.name or
                    dados.get("k") != k or
                    dados.get("method") != "multistart_tabu_multiexchange" or
                    dados.get("status") != "viavel" or
                    dados.get("valida") is False):
                continue
            if dados.get("instance_sha256") not in (None, hash_instancia):
                continue
            solucao = _ler_solucao_json(path, instancia, k)
            custo = avaliar_salas(instancia, solucao)
            melhores.append((custo, path, solucao))
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            continue
    if not melhores:
        return None
    custo, path, solucao = min(melhores, key=lambda x: (x[0], str(x[1])))
    return solucao, custo, str(path.relative_to(ROOT))


def _construir_start(instancia, k: int, seed: int, segundos: float):
    if segundos <= 0:
        return None
    inicio = time.perf_counter()
    prazo = inicio + segundos
    ciclo = 0
    melhor = None
    melhor_custo = math.inf

    def registrar(solucao):
        nonlocal melhor, melhor_custo
        verificar_salas(instancia, solucao, k)
        custo = avaliar_salas(instancia, solucao)
        if custo < melhor_custo:
            melhor, melhor_custo = solucao, custo

    while time.perf_counter() < prazo:
        ciclo += 1
        inicial = construir(instancia, k, seed + ciclo * 1009,
                            reinicios=1, prazo=prazo)
        if inicial.solucao is None:
            continue
        atual = inicial.solucao
        registrar(atual)

        restante = prazo - time.perf_counter()
        tempo_tabu = min(10.0, max(0.0, restante * 0.65))
        if tempo_tabu > 0.05:
            resultado_tabu = busca_tabu(
                instancia, atual, k, seed + ciclo * 1009,
                tempo_limite=tempo_tabu,
                tempo_sem_melhoria=min(5.0, tempo_tabu),
            )
            atual = resultado_tabu.solucao
            registrar(atual)

        restante = prazo - time.perf_counter()
        if restante > 0.05:
            resultado_exchange = busca_multiexchange(
                instancia, atual, k, tempo_limite=restante)
            registrar(resultado_exchange.solucao)

    if melhor is None:
        return None
    return melhor, melhor_custo, "construção + tabu + multi-exchange"


def _adicionar_start(modelo, x, y, z, instancia, k: int, solucao: SolucaoSalas):
    verificar_salas(instancia, solucao, k)
    n, m = len(instancia.candidatos), len(instancia.escolas)
    tipos = tuple(dict.fromkeys(c.tipo_prova for c in instancia.candidatos))
    tipo_sala = {}
    tipos_escola = [set() for _ in instancia.escolas]
    for i, (j, r) in enumerate(solucao.alocacao):
        tipo = instancia.candidatos[i].tipo_prova
        tipo_sala[(j, r)] = tipo
        tipos_escola[j].add(tipo)
    for i in range(n):
        escola = solucao.alocacao[i][0]
        for j in range(m):
            x[i, j].Start = float(j == escola)
    for j, escola in enumerate(instancia.escolas):
        for r in range(len(escola.capacidades_salas)):
            tipo_usado = tipo_sala.get((j, r))
            for tipo in tipos:
                y[j, r, tipo].Start = float(tipo == tipo_usado)
        for tipo in tipos:
            z[j, tipo].Start = float(tipo in tipos_escola[j])


def _numero_finito(valor):
    try:
        valor = float(valor)
    except (TypeError, ValueError, OverflowError):
        return None
    return valor if math.isfinite(valor) else None


def _status_nome(gp, codigo: int) -> str:
    nomes = {
        gp.GRB.OPTIMAL: "OTIMO",
        gp.GRB.INFEASIBLE: "INVIAVEL",
        gp.GRB.INF_OR_UNBD: "INVIAVEL_OU_ILIMITADO",
        gp.GRB.UNBOUNDED: "ILIMITADO",
        gp.GRB.TIME_LIMIT: "LIMITE_TEMPO",
        gp.GRB.INTERRUPTED: "INTERROMPIDO",
        gp.GRB.SUBOPTIMAL: "SUBOTIMO",
        gp.GRB.NUMERIC: "ERRO_NUMERICO",
        gp.GRB.NODE_LIMIT: "LIMITE_NOS",
        gp.GRB.SOLUTION_LIMIT: "LIMITE_SOLUCOES",
    }
    return nomes.get(codigo, f"STATUS_{codigo}")


def _extrair_salas(instancia, k: int, x, y, tipos) -> SolucaoSalas:
    alocacao = [None] * len(instancia.candidatos)
    grupos = {}
    for i, candidato in enumerate(instancia.candidatos):
        valores = [(x[i, j].X, j) for j in range(len(instancia.escolas))]
        melhor_valor, j = max(valores)
        if melhor_valor < 0.5:
            raise RuntimeError(f"O solver não atribuiu o candidato {candidato.id}.")
        grupos.setdefault((j, candidato.tipo_prova), []).append(i)

    for (j, tipo), candidatos in grupos.items():
        salas = [
            r for r in range(len(instancia.escolas[j].capacidades_salas))
            if y[j, r, tipo].X > 0.5
        ]
        salas.sort(key=lambda r: (-instancia.escolas[j].capacidades_salas[r], r))
        cursor = 0
        for r in salas:
            capacidade = instancia.escolas[j].capacidades_salas[r]
            for _ in range(min(capacidade, len(candidatos) - cursor)):
                alocacao[candidatos[cursor]] = (j, r)
                cursor += 1
            if cursor == len(candidatos):
                break
        if cursor != len(candidatos):
            raise RuntimeError(
                f"Capacidade MIP insuficiente ao reconstruir escola {j}, tipo {tipo}."
            )
    if any(posicao is None for posicao in alocacao):
        raise RuntimeError("A reconstrução deixou candidatos sem sala.")
    solucao = SolucaoSalas(tuple(alocacao))
    verificar_salas(instancia, solucao, k)
    return solucao


def _resolver_com_modelo(gp, instancia, k, tipos, indices_por_tipo, *,
                         time_limit, mip_gap, mip_focus, threads, seed, start_info,
                         export_mps, quiet, total_inicio, warm_start_seconds):
    n, m = len(instancia.candidatos), len(instancia.escolas)
    model_name = "alocacao_por_salas"
    with gp.Env() as env:
        build_inicio = time.perf_counter()
        with gp.Model(model_name, env=env) as model:
            model.Params.TimeLimit = time_limit
            model.Params.MIPGap = mip_gap
            model.Params.MIPFocus = mip_focus
            model.Params.Threads = threads
            model.Params.Seed = seed
            model.Params.OutputFlag = 0 if quiet else 1

            x = model.addVars(n, m, vtype=gp.GRB.BINARY, name="x")
            quartos = [(j, r) for j, escola in enumerate(instancia.escolas)
                       for r in range(len(escola.capacidades_salas))]
            y_keys = [(j, r, tipo) for j, r in quartos for tipo in tipos]
            y = model.addVars(y_keys, vtype=gp.GRB.BINARY, name="y")
            z = model.addVars(m, tipos, vtype=gp.GRB.BINARY, name="z")

            model.setObjective(
                gp.quicksum(
                    float(instancia.matriz_distancias[i][j]) * x[i, j]
                    for i in range(n) for j in range(m)
                ), gp.GRB.MINIMIZE,
            )

            for i in range(n):
                model.addConstr(gp.quicksum(x[i, j] for j in range(m)) == 1,
                                name=f"atribuir_{i}")
                tipo = instancia.candidatos[i].tipo_prova
                for j in range(m):
                    model.addConstr(x[i, j] <= z[j, tipo],
                                    name=f"ativar_tipo_{i}_{j}")
            for j, r in quartos:
                model.addConstr(gp.quicksum(y[j, r, tipo] for tipo in tipos) <= 1,
                                name=f"um_tipo_sala_{j}_{r}")
            for j, escola in enumerate(instancia.escolas):
                for tipo in tipos:
                    candidatos = indices_por_tipo[tipo]
                    contagem = gp.quicksum(x[i, j] for i in candidatos)
                    capacidade_tipo = gp.quicksum(
                        escola.capacidades_salas[r] * y[j, r, tipo]
                        for r in range(len(escola.capacidades_salas))
                    )
                    model.addConstr(contagem <= capacidade_tipo,
                                    name=f"capacidade_tipo_{j}_{tipo}")
                    model.addConstr(z[j, tipo] <= contagem,
                                    name=f"tipo_ativo_inf_{j}_{tipo}")
                    for r in range(len(escola.capacidades_salas)):
                        model.addConstr(y[j, r, tipo] <= z[j, tipo],
                                        name=f"sala_tipo_ativo_{j}_{r}_{tipo}")
                model.addConstr(gp.quicksum(z[j, tipo] for tipo in tipos) <= k,
                                name=f"limite_tipos_{j}")

            if start_info is not None:
                _adicionar_start(model, x, y, z, instancia, k, start_info[0])

            if export_mps is not None:
                export_mps.parent.mkdir(parents=True, exist_ok=True)
                model.write(str(export_mps))
            build_seconds = time.perf_counter() - build_inicio
            model.optimize()

            status = _status_nome(gp, int(model.Status))
            solution = None
            valid = False
            objective = bound = gap = None
            if model.SolCount:
                solution = _extrair_salas(instancia, k, x, y, tipos)
                objective = avaliar_salas(instancia, solution)
                reported_obj = _numero_finito(model.ObjVal)
                if reported_obj is None or not math.isclose(
                        objective, reported_obj, rel_tol=1e-8, abs_tol=1e-6):
                    raise RuntimeError(
                        "O custo reavaliado pela matriz não confere com o objetivo do Gurobi."
                    )
                valid = True
            try:
                bound = _numero_finito(model.ObjBound)
            except (AttributeError, gp.GurobiError):
                pass
            if model.SolCount:
                try:
                    gap = _numero_finito(model.MIPGap)
                except (AttributeError, gp.GurobiError):
                    pass

            salas_saida = None
            if solution is not None:
                membros = [[[] for _ in e.capacidades_salas] for e in instancia.escolas]
                for i, (j, r) in enumerate(solution.alocacao):
                    membros[j][r].append(i)
                salas_saida = []
                for j, escola in enumerate(instancia.escolas):
                    for r, capacidade in enumerate(escola.capacidades_salas):
                        indices = membros[j][r]
                        tipo = (instancia.candidatos[indices[0]].tipo_prova
                                if indices else None)
                        salas_saida.append({
                            "escola_id": escola.id,
                            "indice_escola": j,
                            "indice_sala": r,
                            "capacidade": capacidade,
                            "tipo_prova": tipo,
                            "candidatos": [instancia.candidatos[i].id for i in indices],
                        })

            solver_seconds = _numero_finito(getattr(model, "Runtime", None))
            result = {
                "metodo": "mip_gurobi_compacto",
                "formulacao": "vinculo_individual_candidato_tipo",
                "status": status,
                "viavel": valid,
                "instancia": None,
                "k": k,
                "unidade_distancias": instancia.unidade_distancias,
                "objetivo": objective,
                "limite_inferior": bound,
                "gap": gap,
                "solucoes_incumbentes": int(model.SolCount),
                "tempo_solver_s": solver_seconds,
                "tempo_modelagem_s": build_seconds,
                "tempo_total_s": time.perf_counter() - total_inicio,
                "gurobi_versao": ".".join(str(v) for v in gp.gurobi.version()),
                "parametros": {
                    "time_limit_s": time_limit,
                    "mip_gap": mip_gap,
                    "mip_focus": mip_focus,
                    "threads": threads,
                    "seed": seed,
                    "warm_start": start_info[2] if start_info is not None else None,
                    "warm_start_custo": start_info[1] if start_info is not None else None,
                    "warm_start_tempo_s": warm_start_seconds,
                },
                "salas": salas_saida,
                "alocacao": ([list(p) for p in solution.alocacao]
                             if solution is not None else None),
            }
            return result


def _resolver_uma(instancia_path: Path, k: int, args):
    total_inicio = time.perf_counter()
    instancia = ler_instancia(instancia_path)
    warm_start_inicio = time.perf_counter()
    start_info = None
    if args.start_json:
        solucao = _ler_solucao_json(args.start_json, instancia, k)
        start_info = (solucao, avaliar_salas(instancia, solucao), str(args.start_json))
    elif args.warm_start == "auto":
        start_info = _melhor_start_salvo(instancia, k, instancia_path)
        if start_info is None:
            start_info = _construir_start(
                instancia, k, args.seed, args.warm_start_seconds)
    elif args.warm_start == "heuristica":
        start_info = _construir_start(
            instancia, k, args.seed, args.warm_start_seconds)
    warm_start_seconds = time.perf_counter() - warm_start_inicio

    result = _resolver_com_modelo(
        _gurobi(), instancia, k,
        tuple(dict.fromkeys(c.tipo_prova for c in instancia.candidatos)),
        {t: [i for i, c in enumerate(instancia.candidatos) if c.tipo_prova == t]
         for t in dict.fromkeys(c.tipo_prova for c in instancia.candidatos)},
        time_limit=args.time_limit, mip_gap=args.mip_gap, threads=args.threads,
        mip_focus=args.mip_focus,
        seed=args.seed, start_info=start_info,
        export_mps=(args.saida / "mps" / f"{instancia_path.stem}_k{k}.mps"
                    if args.exportar_mps else None),
        quiet=args.quiet, total_inicio=total_inicio,
        warm_start_seconds=warm_start_seconds,
    )
    result["instancia"] = str(instancia_path)
    result["sha256_instancia"] = _sha256(instancia_path)
    destino = args.saida / f"{instancia_path.stem}_k{k}.json"
    destino.parent.mkdir(parents=True, exist_ok=True)
    temporario = destino.with_suffix(destino.suffix + ".tmp")
    temporario.write_text(json.dumps(result, ensure_ascii=False, indent=2,
                                     allow_nan=False) + "\n", encoding="utf-8")
    temporario.replace(destino)
    result["arquivo_resultado"] = str(destino)
    print(
        f"{instancia_path.name} K={k} status={result['status']} "
        f"objetivo={result['objetivo']} bound={result['limite_inferior']} "
        f"gap={result['gap']} tempo={result['tempo_solver_s']}s"
    )
    return result


def _argumentos(argv=None):
    parser = argparse.ArgumentParser(
        description="Resolve instâncias de alocação por sala com Gurobi WLS."
    )
    parser.add_argument("--verificar-licenca", action="store_true",
                        help="confere se o Gurobi encontra uma licença válida")
    origem = parser.add_mutually_exclusive_group()
    origem.add_argument("--instancia", type=Path,
                        help="arquivo de instância para resolver")
    origem.add_argument("--todas-instancias", action="store_true",
                        help="resolve todos os arquivos de data/instancias")
    parser.add_argument("--tipos-por-local", choices=("1", "2", "ambos"),
                        default="1", help="K=1, K=2 ou executar os dois valores")
    parser.add_argument("--time-limit", type=float, default=300.0,
                        help="limite do solver, em segundos, por instância e K")
    parser.add_argument("--mip-gap", type=float, default=0.0,
                        help="gap relativo de parada; 0 tenta provar ótimo")
    parser.add_argument("--mip-focus", type=int, choices=(0, 1, 2, 3), default=0,
                        help="foco do MIP: 0 equilibrado, 3 prioriza melhorar o limite inferior")
    parser.add_argument("--threads", type=int, default=0,
                        help="threads do Gurobi; 0 deixa o solver decidir")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warm-start", choices=("auto", "heuristica", "none"),
                        default="auto",
                        help="usa solução salva, gera uma ou desativa o MIP start")
    parser.add_argument("--warm-start-seconds", type=float, default=30.0,
                        help="orçamento para construir um start quando não há resultado salvo")
    parser.add_argument("--start-json", type=Path,
                        help="arquivo JSON com solução por candidato (escola, sala)")
    parser.add_argument("--exportar-mps", action="store_true",
                        help="também exporta cada modelo no formato MPS")
    parser.add_argument(
        "--saida", type=Path,
        default=ROOT / "resultados" / "exatos" / "candidatos" / "mip",
    )
    parser.add_argument("--quiet", action="store_true",
                        help="não imprime o log detalhado do solver")
    return parser.parse_args(argv)


def main(argv=None):
    args = _argumentos(argv)
    try:
        if args.verificar_licenca:
            print(f"Licença Gurobi validada; versão {verificar_licenca()}.")
            return 0
        if not args.instancia and not args.todas_instancias:
            raise ValueError("Informe --instancia ou --todas-instancias.")
        if args.start_json and (args.todas_instancias or args.tipos_por_local == "ambos"):
            raise ValueError("--start-json só pode ser usado em uma única instância e valor de K.")
        if args.time_limit <= 0 or not math.isfinite(args.time_limit):
            raise ValueError("--time-limit deve ser finito e positivo.")
        if not 0 <= args.mip_gap <= 1 or not math.isfinite(args.mip_gap):
            raise ValueError("--mip-gap deve ficar entre 0 e 1.")
        if args.threads < 0:
            raise ValueError("--threads deve ser zero ou positivo.")
        if args.warm_start_seconds < 0 or not math.isfinite(args.warm_start_seconds):
            raise ValueError("--warm-start-seconds deve ser finito e não negativo.")
        ks = (1, 2) if args.tipos_por_local == "ambos" else (int(args.tipos_por_local),)
        if args.todas_instancias:
            paths = sorted((ROOT / "data" / "instancias").glob("*.txt"))
            if not paths:
                raise ValueError("Nenhuma instância foi encontrada em data/instancias.")
        else:
            paths = [args.instancia.resolve()]
            if not paths[0].is_file():
                raise ValueError(f"Instância não encontrada: {paths[0]}")
        resultados = []
        for path in paths:
            for k in ks:
                resultados.append(_resolver_uma(path, k, args))
        if len(resultados) > 1:
            args.saida.mkdir(parents=True, exist_ok=True)
            campos = ("instancia", "k", "status", "viavel", "objetivo",
                      "limite_inferior", "gap", "tempo_solver_s", "tempo_total_s")
            resumo = args.saida / "resumo.csv"
            with resumo.open("w", newline="", encoding="utf-8") as arquivo:
                escritor = csv.DictWriter(arquivo, fieldnames=campos)
                escritor.writeheader()
                escritor.writerows({campo: resultado.get(campo) for campo in campos}
                                   for resultado in resultados)
            print(f"Resumo salvo em {resumo}")
        return 0 if all(resultado["viavel"] for resultado in resultados) else 2
    except (OSError, ValueError, RuntimeError) as erro:
        print(f"Erro: {erro}", file=sys.stderr)
        return 2
    except Exception as erro:
        gp = sys.modules.get("gurobipy")
        gurobi_error = getattr(gp, "GurobiError", ()) if gp is not None else ()
        if gurobi_error and isinstance(erro, gurobi_error):
            print(
                "Erro do Gurobi. Confira a conexão e a licença WLS em ~/gurobi.lic "
                "ou configure GRB_LICENSE_FILE. Detalhe: " + str(erro),
                file=sys.stderr,
            )
            return 2
        raise


if __name__ == "__main__":
    raise SystemExit(main())
