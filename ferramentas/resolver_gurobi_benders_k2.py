"""Resolvedor Benders por sala para alocação com até dois tipos por escola."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.instancias import ler_instancia
from src.otimizacao.salas import SolucaoSalas, avaliar_salas, verificar_salas
from ferramentas.resolver_gurobi import _gurobi, _ler_solucao_json
from ferramentas.resolver_gurobi_benders import (
    _criar_dual_magnanti_wong,
    _criar_subproblema,
    _finito,
    _resolver_corte_magnanti_wong,
    _resolver_subproblema,
    _salas_json,
    _status_nome,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _padroes_start(instancia, solucao, tipos, y_keys, z_keys):
    y = {key: 0.0 for key in y_keys}
    z = {key: 0.0 for key in z_keys}
    tipo_sala = {}
    for i, (j, r) in enumerate(solucao.alocacao):
        tipo = instancia.candidatos[i].tipo_prova
        anterior = tipo_sala.get((j, r))
        if anterior is not None and anterior != tipo:
            raise ValueError(f"A solução inicial mistura tipos na sala {j}/{r}.")
        tipo_sala[j, r] = tipo
        y[j, r, tipo] = 1.0
        z[j, tipo] = 1.0
    verificar_salas(instancia, solucao, 2)
    return y, z


def _w_por_tipo(instancia, y_pattern, tipos):
    resultado = {}
    for j, escola in enumerate(instancia.escolas):
        for tipo in tipos:
            capacidade_tipo = sum(
                escola.capacidades_salas[r] * y_pattern.get((j, r, tipo), 0.0)
                for r in range(len(escola.capacidades_salas))
            )
            resultado[j, tipo] = capacidade_tipo / escola.capacidade
    return resultado


def _reconstruir_k2(instancia, escola_por_candidato, y_pattern):
    grupos = {}
    for i, j in escola_por_candidato.items():
        tipo = instancia.candidatos[i].tipo_prova
        grupos.setdefault((j, tipo), []).append(i)

    alocacao = [None] * len(instancia.candidatos)
    for (j, tipo), candidatos in grupos.items():
        salas = [
            r for r in range(len(instancia.escolas[j].capacidades_salas))
            if y_pattern.get((j, r, tipo), 0.0) > 0.5
        ]
        salas.sort(key=lambda r: (-instancia.escolas[j].capacidades_salas[r], r))
        cursor = 0
        for r in salas:
            capacidade = instancia.escolas[j].capacidades_salas[r]
            for i in candidatos[cursor:cursor + capacidade]:
                alocacao[i] = (j, r)
            cursor += min(capacidade, len(candidatos) - cursor)
            if cursor == len(candidatos):
                break
        if cursor != len(candidatos):
            raise RuntimeError(
                f"Capacidade de salas do tipo {tipo} insuficiente na escola {j}."
            )
    if any(p is None for p in alocacao):
        raise RuntimeError("A reconstrução K=2 deixou candidatos sem sala.")
    solucao = SolucaoSalas(tuple(alocacao))
    verificar_salas(instancia, solucao, 2)
    return solucao


def resolver(instancia_path: Path, args):
    gp = _gurobi()
    instancia = ler_instancia(instancia_path)
    candidatos_por_tipo = {}
    for i, candidato in enumerate(instancia.candidatos):
        candidatos_por_tipo.setdefault(candidato.tipo_prova, []).append(i)
    tipos = tuple(candidatos_por_tipo)
    n, m = len(instancia.candidatos), len(instancia.escolas)
    capacidades_escola = [escola.capacidade for escola in instancia.escolas]
    quartos = [(j, r) for j, escola in enumerate(instancia.escolas)
               for r in range(len(escola.capacidades_salas))]
    y_keys = [(j, r, tipo) for j, r in quartos for tipo in tipos]
    z_keys = [(j, tipo) for j in range(m) for tipo in tipos]

    start_solution = (_ler_solucao_json(args.start_json, instancia, 2)
                      if args.start_json else None)
    inicio = time.perf_counter()
    deadline = inicio + args.time_limit
    melhor_objetivo = math.inf
    melhor_solucao = None
    limite_inferior = -math.inf
    cortes_adicionados = 0
    cortes_raiz = 0
    cortes_mw = 0
    rodadas_raiz = 0
    iteracoes = 0
    core_y = None
    assinaturas = set()
    status_final = "LIMITE_TEMPO"
    slice_atual = args.master_slice

    with gp.Env() as env:
        model_start = time.perf_counter()
        master = gp.Model("benders_mestre_k2_por_sala", env=env)
        master.Params.OutputFlag = 0
        master.Params.Threads = args.threads
        master.Params.Seed = args.seed
        master.Params.MIPGap = 0.0
        master.Params.MIPFocus = args.mip_focus

        y = master.addVars(y_keys, vtype=gp.GRB.BINARY, name="y")
        z = master.addVars(z_keys, vtype=gp.GRB.BINARY, name="z")
        theta = master.addVars(tipos, lb=0.0, vtype=gp.GRB.CONTINUOUS,
                               name="theta")
        master.setObjective(gp.quicksum(theta[t] for t in tipos), gp.GRB.MINIMIZE)

        # Cada sala recebe no máximo um tipo; escola recebe no máximo dois.
        for j, r in quartos:
            master.addConstr(gp.quicksum(y[j, r, t] for t in tipos) <= 1,
                             name=f"um_tipo_sala_{j}_{r}")
        for j, escola in enumerate(instancia.escolas):
            master.addConstr(gp.quicksum(z[j, t] for t in tipos) <= 2,
                             name=f"limite_tipos_{j}")
            for tipo in tipos:
                salas_tipo = gp.quicksum(
                    y[j, r, tipo] for r in range(len(escola.capacidades_salas))
                )
                master.addConstr(z[j, tipo] <= salas_tipo,
                                 name=f"tipo_tem_sala_{j}_{tipo}")
                for r in range(len(escola.capacidades_salas)):
                    master.addConstr(y[j, r, tipo] <= z[j, tipo],
                                     name=f"sala_requer_tipo_{j}_{r}_{tipo}")
        for tipo in tipos:
            master.addConstr(
                gp.quicksum(
                    instancia.escolas[j].capacidades_salas[r] * y[j, r, tipo]
                    for j, r in quartos
                ) >= len(candidatos_por_tipo[tipo]),
                name=f"capacidade_total_{tipo}",
            )

        subproblemas = {
            tipo: _criar_subproblema(
                gp, env, instancia, tipo, candidatos_por_tipo[tipo], 0)
            for tipo in tipos
        }
        duais_mw = ({
            tipo: _criar_dual_magnanti_wong(
                gp, env, instancia, tipo, candidatos_por_tipo[tipo])
            for tipo in tipos
        } if (args.cortes_magnanti_wong or args.cortes_magnanti_wong_raiz) else {})
        master.update()
        model_build_seconds = time.perf_counter() - model_start

        def normalized_capacity(y_pattern):
            return _w_por_tipo(instancia, y_pattern, tipos)

        def info_cut_for_master(info, tipo):
            room_coeff = []
            for j, r, t in y_keys:
                if t != tipo:
                    room_coeff.append(0.0)
                else:
                    room_coeff.append(
                        info["coeficientes"][j]
                        * instancia.escolas[j].capacidades_salas[r]
                        / capacidades_escola[j]
                    )
            return room_coeff

        def adicionar_corte(tipo, info, nome):
            nonlocal cortes_adicionados
            room_coeff = info_cut_for_master(info, tipo)
            assinatura = (tipo, round(info["constante"], 7),
                          *(round(v, 7) for v in room_coeff))
            if assinatura in assinaturas:
                return False
            expr = info["constante"] + gp.quicksum(
                room_coeff[k] * y[key]
                for k, key in enumerate(y_keys) if key[2] == tipo
            )
            master.addConstr(theta[tipo] >= expr,
                             name=f"corte_{nome}_{tipo}_{cortes_adicionados}")
            assinaturas.add(assinatura)
            cortes_adicionados += 1
            return True

        def selecionar_corte(tipo, w_pattern, info, *, na_raiz=False):
            nonlocal cortes_mw
            usar_mw = (args.cortes_magnanti_wong or
                       (args.cortes_magnanti_wong_raiz and na_raiz))
            if not usar_mw or core_y is None:
                return info
            w_core = normalized_capacity(core_y)
            mw = _resolver_corte_magnanti_wong(
                gp, duais_mw[tipo], instancia, w_pattern, w_core,
                info["objetivo"], deadline)
            if mw is None:
                return None
            selecionado = dict(info)
            selecionado.update(mw)
            cortes_mw += 1
            return selecionado

        def atualizar_corepoint(y_pattern):
            nonlocal core_y
            if not (args.cortes_magnanti_wong or args.cortes_magnanti_wong_raiz):
                return
            if core_y is None:
                if y_start is not None:
                    core_y = {key: 0.5 * y_start[key] + 0.5 * y_pattern.get(key, 0.0)
                              for key in y_keys}
                else:
                    core_y = {key: y_pattern.get(key, 0.0) for key in y_keys}
                return
            core_y = {key: 0.9 * core_y[key] + 0.1 * y_pattern.get(key, 0.0)
                      for key in y_keys}

        def avaliar_padrao(w_pattern, y_pattern, prefixo, *,
                           reconstruir=True, adicionar=True, na_raiz=False):
            infos = {}
            alocacao_escola = {}
            valor_total = 0.0
            for tipo in tipos:
                info = _resolver_subproblema(
                    gp, subproblemas[tipo], instancia, w_pattern, None, deadline,
                    extrair_alocacao=reconstruir)
                if info is None:
                    return None
                info = selecionar_corte(tipo, w_pattern, info, na_raiz=na_raiz)
                if info is None:
                    return None
                infos[tipo] = info
                valor_total += info["objetivo"]
                if reconstruir:
                    alocacao_escola.update(info["alocacao"])
                if adicionar:
                    adicionar_corte(tipo, info, prefixo)
            if not reconstruir:
                return infos, None, valor_total
            solucao = _reconstruir_k2(instancia, alocacao_escola, y_pattern)
            objetivo = avaliar_salas(instancia, solucao)
            if not math.isclose(objetivo, valor_total, rel_tol=1e-8, abs_tol=1e-5):
                raise RuntimeError(
                    f"Custo reconstruído {objetivo:.9f} difere do fluxo "
                    f"{valor_total:.9f}."
                )
            return infos, solucao, objetivo

        y_start = z_start = None
        if start_solution is not None:
            y_start, z_start = _padroes_start(
                instancia, start_solution, tipos, y_keys, z_keys)

        def separar_raiz():
            """Separa cortes de Benders na relaxação da raiz."""
            nonlocal cortes_raiz, rodadas_raiz, limite_inferior, core_y
            for _ in range(args.max_rodadas_raiz):
                if time.perf_counter() >= deadline:
                    return False
                relaxado = master.relax()
                try:
                    relaxado.Params.OutputFlag = 0
                    relaxado.Params.Threads = 1
                    relaxado.Params.Method = 1
                    relaxado.Params.TimeLimit = max(0.001, deadline - time.perf_counter())
                    relaxado.optimize()
                    if relaxado.Status != gp.GRB.OPTIMAL:
                        return False
                    limite_inferior = max(limite_inferior, float(relaxado.ObjVal))
                    y_root = {key: float(relaxado.getVarByName(y[key].VarName).X)
                              for key in y_keys}
                    if (args.cortes_magnanti_wong or args.cortes_magnanti_wong_raiz) and core_y is None:
                        if y_start is not None:
                            core_y = {key: 0.5 * y_start[key] + 0.5 * y_root[key]
                                      for key in y_keys}
                        else:
                            core_y = dict(y_root)
                    w_root = normalized_capacity(y_root)
                    theta_root = {t: float(relaxado.getVarByName(theta[t].VarName).X)
                                  for t in tipos}
                    before = cortes_adicionados
                    infos = avaliar_padrao(
                        w_root, y_root, f"raiz_{rodadas_raiz}",
                        reconstruir=False, adicionar=False, na_raiz=True)
                    if infos is None:
                        return False
                    for tipo, info in infos[0].items():
                        if theta_root[tipo] < info["objetivo"] - 1e-5:
                            if adicionar_corte(tipo, info, f"raiz_{rodadas_raiz}"):
                                cortes_raiz += 1
                    master.update()
                    rodadas_raiz += 1
                    new_cuts = cortes_adicionados - before
                    print(f"Raiz {rodadas_raiz}: LB-LP={relaxado.ObjVal:.6f} "
                          f"novos_cortes={new_cuts} total_raiz={cortes_raiz}")
                    if new_cuts == 0:
                        return True
                finally:
                    relaxado.dispose()
            return True

        # All rooms available to each type gives a valid recourse lower bound.
        all_open_w = {(j, tipo): 1.0 for j in range(m) for tipo in tipos}
        all_open_y = {key: 1.0 for key in y_keys}
        base = avaliar_padrao(all_open_w, all_open_y, "base",
                              reconstruir=False, adicionar=True, na_raiz=False)
        if base is None:
            infeasible = any(sub["model"].Status == gp.GRB.INFEASIBLE
                             for sub in subproblemas.values())
            status_final = "INVIAVEL" if infeasible else "LIMITE_TEMPO"
        else:
            for tipo, info in base[0].items():
                theta[tipo].LB = info["objetivo"]
            master.update()

            if start_solution is not None and time.perf_counter() < deadline:
                w_start = normalized_capacity(y_start)
                start_eval = avaliar_padrao(w_start, y_start, "start")
                if start_eval is not None:
                    start_infos, start_sol, start_obj = start_eval
                    melhor_solucao, melhor_objetivo = start_sol, start_obj
                    for key in y_keys:
                        y[key].Start = y_start[key]
                    for key in z_keys:
                        z[key].Start = z_start[key]
                    for tipo in tipos:
                        theta[tipo].Start = start_infos[tipo]["objetivo"]
                    master.update()

            can_continue = True
            if args.cortes_raiz_inicial:
                can_continue = separar_raiz()
                status_final = ("EM_ANDAMENTO" if can_continue
                                and time.perf_counter() < deadline else "LIMITE_TEMPO")

            while can_continue and time.perf_counter() < deadline:
                iteracoes += 1
                if args.cortes_raiz:
                    if not separar_raiz() or time.perf_counter() >= deadline:
                        status_final = "LIMITE_TEMPO"
                        break
                remaining = deadline - time.perf_counter()
                master.Params.TimeLimit = max(0.001, min(slice_atual, remaining))
                master.optimize()
                try:
                    limite_inferior = max(limite_inferior, float(master.ObjBound))
                except (AttributeError, gp.GurobiError):
                    pass

                if master.Status == gp.GRB.INFEASIBLE:
                    status_final = "INVIAVEL"
                    break
                if master.SolCount == 0:
                    status_final = ("LIMITE_TEMPO_SEM_INCUMBENTE_MESTRE"
                                    if master.Status == gp.GRB.TIME_LIMIT
                                    else _status_nome(gp, int(master.Status)))
                    break

                y_pattern = {key: float(y[key].X > 0.5) for key in y_keys}
                z_pattern = {key: float(z[key].X > 0.5) for key in z_keys}
                w_pattern = normalized_capacity(y_pattern)
                cuts_before = cortes_adicionados
                evaluated = avaliar_padrao(w_pattern, y_pattern,
                                           f"inteira_{iteracoes}")
                if evaluated is None:
                    status_final = "LIMITE_TEMPO"
                    break
                infos, solucao, objective = evaluated
                if objective < melhor_objetivo - 1e-7:
                    melhor_objetivo, melhor_solucao = objective, solucao
                atualizar_corepoint(y_pattern)

                bound = _finito(limite_inferior)
                gap = (max(0.0, (melhor_objetivo - bound)
                           / max(abs(melhor_objetivo), 1e-12))
                       if bound is not None and math.isfinite(melhor_objetivo) else None)
                violated = any(theta[t].X < infos[t]["objetivo"] - 1e-5
                               for t in tipos)
                new_cut = cortes_adicionados > cuts_before
                print(f"Iteração {iteracoes}: mestre={_status_nome(gp, int(master.Status))} "
                      f"LB={bound if bound is not None else '—'} "
                      f"UB={melhor_objetivo:.6f} gap={gap if gap is not None else '—'} "
                      f"cortes={cortes_adicionados}")

                for key in y_keys:
                    y[key].Start = y_pattern[key]
                for key in z_keys:
                    z[key].Start = z_pattern[key]
                for tipo in tipos:
                    theta[tipo].Start = infos[tipo]["objetivo"]

                if gap is not None and gap <= args.mip_gap + 1e-10:
                    status_final = "OTIMO" if gap <= 1e-8 else "GAP_ATINGIDO"
                    break
                if (master.Status == gp.GRB.OPTIMAL and not violated
                        and abs(melhor_objetivo - float(master.ObjVal)) <= 1e-5):
                    status_final = "OTIMO"
                    limite_inferior = melhor_objetivo
                    break
                if not new_cut:
                    if (master.Status == gp.GRB.TIME_LIMIT
                            and time.perf_counter() < deadline):
                        slice_atual = deadline - time.perf_counter()
                        continue
                    status_final = "SEM_NOVO_CORTE"
                    break
                status_final = ("LIMITE_TEMPO" if master.Status != gp.GRB.OPTIMAL
                                else "EM_ANDAMENTO")

            if status_final == "EM_ANDAMENTO":
                status_final = "LIMITE_TEMPO"

        master.dispose()
        for sub in subproblemas.values():
            sub["model"].dispose()
        for dual in duais_mw.values():
            dual["model"].dispose()

    if melhor_solucao is not None:
        verificar_salas(instancia, melhor_solucao, 2)
    gap_final = None
    if melhor_solucao is not None and math.isfinite(limite_inferior):
        gap_final = max(0.0, (melhor_objetivo - limite_inferior)
                        / max(abs(melhor_objetivo), 1e-12))
    elapsed = time.perf_counter() - inicio
    return {
        "metodo": "benders_fluxo_por_tipo_k2",
        "formulacao": "mestre_tipo_por_sala_subproblemas_fluxo_por_tipo",
        "status": status_final,
        "viavel": melhor_solucao is not None,
        "instancia": str(instancia_path),
        "sha256_instancia": _sha256(instancia_path),
        "k": 2,
        "unidade_distancias": instancia.unidade_distancias,
        "objetivo": melhor_objetivo if math.isfinite(melhor_objetivo) else None,
        "limite_inferior": _finito(limite_inferior),
        "gap": gap_final,
        "iteracoes_benders": iteracoes,
        "cortes_adicionados": cortes_adicionados,
        "cortes_fracionarios_raiz": cortes_raiz,
        "rodadas_separacao_raiz": rodadas_raiz,
        "cortes_magnanti_wong_gerados": cortes_mw,
        "tempo_modelagem_s": model_build_seconds,
        "tempo_total_s": elapsed,
        "gurobi_versao": ".".join(str(v) for v in gp.gurobi.version()),
        "parametros": {
            "time_limit_s": args.time_limit,
            "mip_gap": args.mip_gap,
            "master_slice_s": args.master_slice,
            "cortes_raiz": args.cortes_raiz,
            "cortes_raiz_inicial": args.cortes_raiz_inicial,
            "max_rodadas_raiz": args.max_rodadas_raiz,
            "cortes_magnanti_wong": args.cortes_magnanti_wong,
            "cortes_magnanti_wong_raiz": args.cortes_magnanti_wong_raiz,
            "mip_focus": args.mip_focus,
            "threads_mestre": args.threads,
            "seed": args.seed,
            "start_json": str(args.start_json) if args.start_json else None,
        },
        "salas": _salas_json(instancia, melhor_solucao) if melhor_solucao else None,
        "alocacao": ([list(p) for p in melhor_solucao.alocacao]
                     if melhor_solucao else None),
    }


def argumentos(argv=None):
    parser = argparse.ArgumentParser(
        description="Decomposição de Benders para alocação de salas com K=2."
    )
    parser.add_argument("--instancia", type=Path, required=True)
    parser.add_argument("--time-limit", type=float, default=600.0)
    parser.add_argument("--mip-gap", type=float, default=0.0)
    parser.add_argument("--master-slice", type=float, default=15.0)
    separacao_raiz = parser.add_mutually_exclusive_group()
    separacao_raiz.add_argument("--cortes-raiz", action="store_true")
    separacao_raiz.add_argument("--cortes-raiz-inicial", action="store_true")
    parser.add_argument("--max-rodadas-raiz", type=int, default=3)
    selecao_mw = parser.add_mutually_exclusive_group()
    selecao_mw.add_argument("--cortes-magnanti-wong", action="store_true")
    selecao_mw.add_argument("--cortes-magnanti-wong-raiz", action="store_true")
    parser.add_argument("--mip-focus", type=int, choices=(0, 1, 2, 3), default=2)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--start-json", type=Path,
                        help="solução K=2 viável para iniciar o incumbente")
    parser.add_argument("--saida", type=Path,
                        default=ROOT / "resultados" / "exatos" / "candidatos" / "benders_k2")
    args = parser.parse_args(argv)
    if not args.instancia.is_file():
        parser.error(f"instância não encontrada: {args.instancia}")
    if args.start_json and not args.start_json.is_file():
        parser.error(f"solução inicial não encontrada: {args.start_json}")
    if args.time_limit <= 0 or not math.isfinite(args.time_limit):
        parser.error("--time-limit deve ser finito e positivo")
    if not 0 <= args.mip_gap <= 1 or not math.isfinite(args.mip_gap):
        parser.error("--mip-gap deve estar entre 0 e 1")
    if args.master_slice <= 0 or not math.isfinite(args.master_slice):
        parser.error("--master-slice deve ser finito e positivo")
    if args.max_rodadas_raiz < 1:
        parser.error("--max-rodadas-raiz deve ser pelo menos 1")
    if args.cortes_magnanti_wong_raiz and not args.cortes_raiz_inicial:
        parser.error("--cortes-magnanti-wong-raiz exige --cortes-raiz-inicial")
    if args.threads < 0:
        parser.error("--threads deve ser zero ou positivo")
    return args


def main(argv=None):
    args = argumentos(argv)
    try:
        result = resolver(args.instancia.resolve(), args)
        destino = args.saida / f"{args.instancia.stem}_k2.json"
        destino.parent.mkdir(parents=True, exist_ok=True)
        temporario = destino.with_suffix(destino.suffix + ".tmp")
        temporario.write_text(json.dumps(
            result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        temporario.replace(destino)
        print(
            f"{args.instancia.name} K=2 status={result['status']} "
            f"objetivo={result['objetivo']} bound={result['limite_inferior']} "
            f"gap={result['gap']} tempo={result['tempo_total_s']:.2f}s"
        )
        print(f"Resultado salvo em {destino}")
        return 0 if result["viavel"] or result["status"] == "INVIAVEL" else 2
    except (OSError, ValueError, RuntimeError) as erro:
        print(f"Erro: {erro}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
