"""Resolvedor de decomposição de Benders para alocação com K=1."""
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


def _status_nome(gp, status: int) -> str:
    return {
        gp.GRB.OPTIMAL: "OTIMO",
        gp.GRB.INFEASIBLE: "INVIAVEL",
        gp.GRB.INF_OR_UNBD: "INVIAVEL_OU_ILIMITADO",
        gp.GRB.TIME_LIMIT: "LIMITE_TEMPO",
        gp.GRB.INTERRUPTED: "INTERROMPIDO",
        gp.GRB.SUBOPTIMAL: "SUBOTIMO",
    }.get(status, f"STATUS_{status}")


def _finito(valor):
    try:
        valor = float(valor)
    except (TypeError, ValueError, OverflowError):
        return None
    return valor if math.isfinite(valor) else None


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _padrao_tipos(instancia, solucao: SolucaoSalas, tipos):
    padrao = {}
    for i, (j, _r) in enumerate(solucao.alocacao):
        tipo = instancia.candidatos[i].tipo_prova
        anterior = padrao.get(j)
        if anterior is not None and anterior != tipo:
            raise ValueError(f"A solução inicial usa mais de um tipo na escola {j}.")
        padrao[j] = tipo
    return {(j, tipo): float(padrao.get(j) == tipo)
            for j in range(len(instancia.escolas)) for tipo in tipos}


def _criar_subproblema(gp, env, instancia, tipo, candidatos, output_flag):
    m = len(instancia.escolas)
    lp = gp.Model(f"transporte_{tipo}", env=env)
    lp.Params.OutputFlag = output_flag
    lp.Params.Threads = 1
    lp.Params.Method = 1  # dual simplex reaproveita a base entre padrões de escolas
    lp.Params.NumericFocus = 1

    x = lp.addVars(len(candidatos), m, lb=0.0, vtype=gp.GRB.CONTINUOUS,
                   name="x")
    demandas = [
        lp.addConstr(gp.quicksum(x[q, j] for j in range(m)) == 1,
                     name=f"demanda_{q}")
        for q in range(len(candidatos))
    ]
    capacidades = [
        lp.addConstr(
            gp.quicksum(x[q, j] for q in range(len(candidatos)))
            <= instancia.escolas[j].capacidade,
            name=f"capacidade_{j}",
        )
        for j in range(m)
    ]
    lp.setObjective(gp.quicksum(
        float(instancia.matriz_distancias[i][j]) * x[q, j]
        for q, i in enumerate(candidatos) for j in range(m)
    ), gp.GRB.MINIMIZE)
    lp.update()
    return {"tipo": tipo, "candidatos": candidatos, "model": lp, "x": x,
            "demandas": demandas, "capacidades": capacidades}


def _resolver_subproblema(gp, sub, instancia, padrao, z, deadline,
                          extrair_alocacao=True):
    lp = sub["model"]
    restante = deadline - time.perf_counter()
    if restante <= 0:
        return None
    m = len(instancia.escolas)
    rhs = [instancia.escolas[j].capacidade * padrao.get((j, sub["tipo"]), 0.0)
           for j in range(m)]
    lp.setAttr(gp.GRB.Attr.RHS, sub["capacidades"], rhs)
    lp.Params.TimeLimit = max(0.001, restante)
    lp.optimize()
    if lp.Status != gp.GRB.OPTIMAL:
        return None

    dual_demanda = [float(c.Pi) for c in sub["demandas"]]
    dual_capacidade = [float(c.Pi) for c in sub["capacidades"]]
    constante = sum(dual_demanda)
    coeficientes = [
        instancia.escolas[j].capacidade * dual_capacidade[j]
        for j in range(m)
    ]
    valor_corte = constante + sum(
        coeficientes[j] * padrao.get((j, sub["tipo"]), 0.0)
        for j in range(m)
    )
    objetivo = float(lp.ObjVal)
    if not math.isclose(valor_corte, objetivo, rel_tol=1e-7, abs_tol=1e-5):
        raise RuntimeError(
            f"Corte dual inconsistente em {sub['tipo']}: "
            f"dual={valor_corte:.9f}, primal={objetivo:.9f}."
        )

    alocacao_tipo = None
    if extrair_alocacao:
        alocacao_tipo = {}
        x = sub["x"]
        for q, i in enumerate(sub["candidatos"]):
            j = max(range(m), key=lambda escola: x[q, escola].X)
            valor = float(x[q, j].X)
            if valor < 1.0 - 1e-6:
                raise RuntimeError(
                    f"O subproblema de {sub['tipo']} retornou fluxo fracionário "
                    f"para o candidato {instancia.candidatos[i].id}: {valor}."
                )
            alocacao_tipo[i] = j
    return {"objetivo": objetivo, "constante": constante,
            "coeficientes": coeficientes, "alocacao": alocacao_tipo}


def _criar_dual_magnanti_wong(gp, env, instancia, tipo, candidatos):
    """Dual explícito usado para escolher cortes Magnanti–Wong."""
    m = len(instancia.escolas)
    modelo = gp.Model(f"dual_mw_{tipo}", env=env)
    modelo.Params.OutputFlag = 0
    modelo.Params.Threads = 1
    modelo.Params.Method = 1
    modelo.Params.NumericFocus = 1

    alfa = modelo.addVars(len(candidatos), lb=-gp.GRB.INFINITY,
                          ub=gp.GRB.INFINITY, name="alpha")
    beta = modelo.addVars(m, lb=-gp.GRB.INFINITY, ub=0.0, name="beta")
    modelo.addConstrs(
        (alfa[q] + beta[j] <= float(instancia.matriz_distancias[i][j])
         for q, i in enumerate(candidatos) for j in range(m)),
        name="dual_viavel",
    )
    face = modelo.addConstr(gp.quicksum(alfa[q] for q in range(len(candidatos)))
                            == 0.0, name="face_otima")
    modelo.setObjective(gp.quicksum(alfa[q] for q in range(len(candidatos))),
                        gp.GRB.MAXIMIZE)
    modelo.update()
    return {"tipo": tipo, "model": modelo, "alpha": alfa, "beta": beta,
            "face": face, "candidatos": candidatos}


def _resolver_corte_magnanti_wong(gp, dual, instancia, padrao, corepoint,
                                  objetivo_subproblema, deadline):
    """Escolhe um dual ótimo que maximiza o corte no ponto core."""
    modelo = dual["model"]
    restante = deadline - time.perf_counter()
    if restante <= 0:
        return None

    m = len(instancia.escolas)
    tipo = dual["tipo"]
    for j, escola in enumerate(instancia.escolas):
        modelo.chgCoeff(
            dual["face"], dual["beta"][j],
            escola.capacidade * padrao.get((j, tipo), 0.0),
        )
    modelo.setAttr(gp.GRB.Attr.RHS, [dual["face"]], [objetivo_subproblema])
    modelo.setAttr(
        gp.GRB.Attr.Obj, list(dual["alpha"].values()),
        [1.0] * len(dual["candidatos"]),
    )
    modelo.setAttr(
        gp.GRB.Attr.Obj, list(dual["beta"].values()),
        [instancia.escolas[j].capacidade * corepoint.get((j, tipo), 0.0)
         for j in range(m)],
    )
    modelo.Params.TimeLimit = max(0.001, restante)
    modelo.optimize()
    if modelo.Status != gp.GRB.OPTIMAL:
        return None

    constante = sum(float(dual["alpha"][q].X)
                    for q in range(len(dual["candidatos"])))
    coeficientes = [
        instancia.escolas[j].capacidade * float(dual["beta"][j].X)
        for j in range(m)
    ]
    valor_no_ponto = constante + sum(
        coeficientes[j] * padrao.get((j, tipo), 0.0) for j in range(m)
    )
    if not math.isclose(valor_no_ponto, objetivo_subproblema,
                        rel_tol=1e-7, abs_tol=1e-5):
        raise RuntimeError(
            f"Corte Magnanti–Wong não está tight em {tipo}: "
            f"dual={valor_no_ponto:.9f}, primal={objetivo_subproblema:.9f}."
        )
    return {"constante": constante, "coeficientes": coeficientes}


def _reconstruir(instancia, alocacao_escola):
    membros = {}
    for i, j in alocacao_escola.items():
        membros.setdefault((j, instancia.candidatos[i].tipo_prova), []).append(i)

    alocacao = [None] * len(instancia.candidatos)
    for (j, _tipo), candidatos in membros.items():
        salas = sorted(
            range(len(instancia.escolas[j].capacidades_salas)),
            key=lambda r: (-instancia.escolas[j].capacidades_salas[r], r),
        )
        cursor = 0
        for r in salas:
            capacidade = instancia.escolas[j].capacidades_salas[r]
            for i in candidatos[cursor:cursor + capacidade]:
                alocacao[i] = (j, r)
            cursor += min(capacidade, len(candidatos) - cursor)
            if cursor == len(candidatos):
                break
        if cursor != len(candidatos):
            raise RuntimeError(f"Capacidade insuficiente ao reconstruir a escola {j}.")
    if any(p is None for p in alocacao):
        raise RuntimeError("A reconstrução deixou candidatos sem sala.")
    solucao = SolucaoSalas(tuple(alocacao))
    verificar_salas(instancia, solucao, 1)
    return solucao


def _salas_json(instancia, solucao):
    membros = [[[] for _ in e.capacidades_salas] for e in instancia.escolas]
    for i, (j, r) in enumerate(solucao.alocacao):
        membros[j][r].append(i)
    salas = []
    for j, escola in enumerate(instancia.escolas):
        for r, capacidade in enumerate(escola.capacidades_salas):
            indices = membros[j][r]
            salas.append({
                "escola_id": escola.id,
                "indice_escola": j,
                "indice_sala": r,
                "capacidade": capacidade,
                "tipo_prova": (instancia.candidatos[indices[0]].tipo_prova
                               if indices else None),
                "candidatos": [instancia.candidatos[i].id for i in indices],
            })
    return salas


def resolver(instancia_path: Path, args):
    gp = _gurobi()
    instancia = ler_instancia(instancia_path)
    candidatos_por_tipo = {}
    for i, candidato in enumerate(instancia.candidatos):
        candidatos_por_tipo.setdefault(candidato.tipo_prova, []).append(i)
    tipos = tuple(candidatos_por_tipo)
    m = len(instancia.escolas)
    capacidades_escola = [e.capacidade for e in instancia.escolas]

    start_solution = None
    if args.start_json:
        start_solution = _ler_solucao_json(args.start_json, instancia, 1)

    inicio = time.perf_counter()
    deadline = inicio + args.time_limit
    tempo_modelagem = 0.0
    cortes_adicionados = 0
    cortes_raiz = 0
    rodadas_raiz = 0
    cortes_mw_gerados = 0
    assinaturas_cortes = set()
    melhor_objetivo = math.inf
    melhor_solucao = None
    limite_inferior = -math.inf
    iteracoes = 0
    status_final = "LIMITE_TEMPO"
    slice_atual = args.master_slice
    corepoint = None

    with gp.Env() as env:
        model_inicio = time.perf_counter()
        master = gp.Model("benders_mestre_k1", env=env)
        master.Params.OutputFlag = 0
        master.Params.Threads = args.threads
        master.Params.Seed = args.seed
        master.Params.MIPGap = 0.0
        master.Params.MIPFocus = args.mip_focus

        z_keys = [(j, tipo) for j in range(m) for tipo in tipos]
        padrao_start = (_padrao_tipos(instancia, start_solution, tipos)
                        if start_solution is not None else None)
        z = master.addVars(z_keys, vtype=gp.GRB.BINARY, name="z")
        theta = master.addVars(tipos, lb=0.0, vtype=gp.GRB.CONTINUOUS,
                               name="theta")
        master.setObjective(gp.quicksum(theta[t] for t in tipos), gp.GRB.MINIMIZE)
        for j in range(m):
            master.addConstr(gp.quicksum(z[j, t] for t in tipos) <= 1,
                             name=f"um_tipo_{j}")
        for tipo in tipos:
            demanda = len(candidatos_por_tipo[tipo])
            master.addConstr(
                gp.quicksum(capacidades_escola[j] * z[j, tipo] for j in range(m))
                >= demanda,
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
        } if (args.cortes_magnanti_wong or
              args.cortes_magnanti_wong_raiz) else {})
        master.update()
        tempo_modelagem = time.perf_counter() - model_inicio

        def adicionar_corte(tipo, info, nome_prefixo):
            nonlocal cortes_adicionados
            nums = [info["constante"], *info["coeficientes"]]
            assinatura = (tipo, *(round(v, 7) for v in nums))
            if assinatura in assinaturas_cortes:
                return False
            expr = info["constante"] + gp.quicksum(
                info["coeficientes"][j] * z[j, tipo] for j in range(m)
            )
            master.addConstr(theta[tipo] >= expr,
                             name=f"{nome_prefixo}_{tipo}_{cortes_adicionados}")
            assinaturas_cortes.add(assinatura)
            cortes_adicionados += 1
            return True

        def selecionar_corte(tipo, padrao, info, na_raiz=False):
            nonlocal cortes_mw_gerados
            aplicar_mw = (args.cortes_magnanti_wong or
                          (args.cortes_magnanti_wong_raiz and na_raiz))
            if not aplicar_mw or corepoint is None:
                return info
            corte_mw = _resolver_corte_magnanti_wong(
                gp, duais_mw[tipo], instancia, padrao, corepoint,
                info["objetivo"], deadline)
            if corte_mw is None:
                return None
            info = dict(info)
            info.update(corte_mw)
            cortes_mw_gerados += 1
            return info

        def atualizar_corepoint(padrao):
            nonlocal corepoint
            if not (args.cortes_magnanti_wong or
                    args.cortes_magnanti_wong_raiz):
                return
            if corepoint is None:
                if padrao_start is not None:
                    corepoint = {
                        key: 0.5 * padrao_start[key] + 0.5 * padrao.get(key, 0.0)
                        for key in z_keys
                    }
                else:
                    corepoint = {key: padrao.get(key, 0.0) for key in z_keys}
                return
            # A combinação convexa permanece factível para as restrições do mestre.
            corepoint = {
                key: 0.9 * corepoint[key] + 0.1 * padrao.get(key, 0.0)
                for key in z_keys
            }

        def avaliar_padrao(padrao, prefixo, reconstruir=True):
            infos = {}
            alocacao_escola = {}
            valor_total = 0.0
            for tipo in tipos:
                info = _resolver_subproblema(
                    gp, subproblemas[tipo], instancia, padrao, z, deadline)
                if info is None:
                    return None
                info = selecionar_corte(tipo, padrao, info)
                if info is None:
                    return None
                infos[tipo] = info
                valor_total += info["objetivo"]
                alocacao_escola.update(info["alocacao"])
                adicionar_corte(tipo, info, prefixo)
            if not reconstruir:
                return infos, None, valor_total
            solucao = _reconstruir(instancia, alocacao_escola)
            objetivo = avaliar_salas(instancia, solucao)
            if not math.isclose(objetivo, valor_total, rel_tol=1e-8, abs_tol=1e-5):
                raise RuntimeError(
                    f"Custo reconstruído {objetivo:.9f} difere do fluxo "
                    f"{valor_total:.9f}."
                )
            return infos, solucao, objetivo

        def separar_raiz():
            """Adiciona cortes de Benders violados pela relaxação do mestre."""
            nonlocal cortes_raiz, rodadas_raiz, limite_inferior, corepoint
            for rodada in range(args.max_rodadas_raiz):
                if time.perf_counter() >= deadline:
                    return False
                relaxado = master.relax()
                try:
                    relaxado.Params.OutputFlag = 0
                    relaxado.Params.Threads = 1
                    relaxado.Params.Method = 1
                    relaxado.Params.TimeLimit = max(
                        0.001, deadline - time.perf_counter())
                    relaxado.optimize()
                    if relaxado.Status != gp.GRB.OPTIMAL:
                        return False

                    limite_inferior = max(limite_inferior, float(relaxado.ObjVal))
                    padrao_fracionario = {
                        key: float(relaxado.getVarByName(z[key].VarName).X)
                        for key in z_keys
                    }
                    if (args.cortes_magnanti_wong or
                            args.cortes_magnanti_wong_raiz) and corepoint is None:
                        if padrao_start is not None:
                            corepoint = {
                                key: 0.5 * padrao_start[key]
                                + 0.5 * padrao_fracionario[key]
                                for key in z_keys
                            }
                        else:
                            corepoint = dict(padrao_fracionario)
                    theta_raiz = {
                        tipo: float(relaxado.getVarByName(theta[tipo].VarName).X)
                        for tipo in tipos
                    }
                    antes = cortes_adicionados
                    for tipo in tipos:
                        info = _resolver_subproblema(
                            gp, subproblemas[tipo], instancia,
                            padrao_fracionario, z, deadline,
                            extrair_alocacao=False)
                        if info is None:
                            return False
                        info = selecionar_corte(
                            tipo, padrao_fracionario, info, na_raiz=True)
                        if info is None:
                            return False
                        if theta_raiz[tipo] < info["objetivo"] - 1e-5:
                            if adicionar_corte(tipo, info, f"corte_raiz_{rodadas_raiz}"):
                                cortes_raiz += 1
                    master.update()
                    rodadas_raiz += 1
                    novos = cortes_adicionados - antes
                    print(
                        f"Raiz {rodadas_raiz}: LB-LP={relaxado.ObjVal:.6f} "
                        f"novos_cortes={novos} total_raiz={cortes_raiz}"
                    )
                    if novos == 0:
                        return True
                finally:
                    relaxado.dispose()
            return True

        # Limites inferiores por tipo: relaxa a escolha exclusiva de escolas,
        # permitindo que todos os locais atendam cada tipo simultaneamente.
        todos_abertos = {(j, t): 1.0 for j in range(m) for t in tipos}
        cortes_base = avaliar_padrao(
            todos_abertos, "corte_base", reconstruir=False)
        if cortes_base is None:
            status_final = "LIMITE_TEMPO"
        else:
            for tipo, info in cortes_base[0].items():
                theta[tipo].LB = info["objetivo"]
            master.update()

            if start_solution is not None and time.perf_counter() < deadline:
                aval_start = avaliar_padrao(padrao_start, "corte_start")
                if aval_start is not None:
                    _infos, solucao_start, objetivo_start = aval_start
                    melhor_solucao = solucao_start
                    melhor_objetivo = objetivo_start
                    for key in z_keys:
                        z[key].Start = padrao_start[key]
                    for tipo in tipos:
                        theta[tipo].Start = _infos[tipo]["objetivo"]
                    master.update()

            pode_continuar = True
            if args.cortes_raiz_inicial:
                pode_continuar = separar_raiz()
                if pode_continuar and time.perf_counter() < deadline:
                    status_final = "EM_ANDAMENTO"
                else:
                    status_final = "LIMITE_TEMPO"

            while pode_continuar and time.perf_counter() < deadline:
                iteracoes += 1
                if args.cortes_raiz:
                    if not separar_raiz():
                        status_final = "LIMITE_TEMPO"
                        break
                    if time.perf_counter() >= deadline:
                        status_final = "LIMITE_TEMPO"
                        break
                restante = deadline - time.perf_counter()
                master.Params.TimeLimit = max(
                    0.001, min(slice_atual, restante))
                master.optimize()
                try:
                    limite_inferior = max(limite_inferior, float(master.ObjBound))
                except (AttributeError, gp.GurobiError):
                    pass

                if master.Status == gp.GRB.INFEASIBLE:
                    status_final = "INVIAVEL"
                    break
                if master.SolCount == 0:
                    if master.Status == gp.GRB.TIME_LIMIT:
                        status_final = "LIMITE_TEMPO_SEM_INCUMBENTE_MESTRE"
                    else:
                        status_final = _status_nome(gp, int(master.Status))
                    break

                padrao = {}
                for j in range(m):
                    ativos = [t for t in tipos if z[j, t].X > 0.5]
                    if len(ativos) > 1:
                        raise RuntimeError(f"Mestre escolheu dois tipos para a escola {j}.")
                    if ativos:
                        padrao[j, ativos[0]] = 1.0

                cortes_antes = cortes_adicionados
                aval = avaliar_padrao(padrao, f"corte_{iteracoes}")
                if aval is None:
                    status_final = "LIMITE_TEMPO"
                    break
                infos, solucao, objetivo = aval
                if objetivo < melhor_objetivo - 1e-7:
                    melhor_objetivo = objetivo
                    melhor_solucao = solucao
                atualizar_corepoint(padrao)

                bound_atual = _finito(limite_inferior)
                gap = (max(0.0, (melhor_objetivo - bound_atual) /
                           max(abs(melhor_objetivo), 1e-12))
                       if bound_atual is not None and math.isfinite(melhor_objetivo)
                       else None)
                violacao = any(
                    theta[t].X < infos[t]["objetivo"] - 1e-5
                    for t in tipos
                )
                novo_corte = cortes_adicionados > cortes_antes
                print(
                    f"Iteração {iteracoes}: mestre={_status_nome(gp, int(master.Status))} "
                    f"LB={bound_atual if bound_atual is not None else '—'} "
                    f"UB={melhor_objetivo:.6f} gap={gap if gap is not None else '—'} "
                    f"cortes={cortes_adicionados}"
                )

                for key in z_keys:
                    z[key].Start = padrao.get(key, 0.0)
                for tipo in tipos:
                    theta[tipo].Start = infos[tipo]["objetivo"]

                if gap is not None and gap <= args.mip_gap + 1e-10:
                    status_final = "OTIMO" if gap <= 1e-8 else "GAP_ATINGIDO"
                    break
                if (master.Status == gp.GRB.OPTIMAL and not violacao and
                        abs(melhor_objetivo - float(master.ObjVal)) <= 1e-5):
                    status_final = "OTIMO"
                    limite_inferior = melhor_objetivo
                    break
                if not novo_corte:
                    if (master.Status == gp.GRB.TIME_LIMIT and
                            time.perf_counter() < deadline):
                        # O padrão atual já está cortado; use o orçamento
                        # restante para a árvore do mestre explorar alternativas.
                        slice_atual = deadline - time.perf_counter()
                        continue
                    status_final = "SEM_NOVO_CORTE"
                    break
                if master.Status != gp.GRB.OPTIMAL:
                    status_final = "LIMITE_TEMPO"
                else:
                    status_final = "EM_ANDAMENTO"

            if status_final == "EM_ANDAMENTO":
                status_final = "LIMITE_TEMPO"

        master.dispose()
        for sub in subproblemas.values():
            sub["model"].dispose()
        for dual in duais_mw.values():
            dual["model"].dispose()

    if melhor_solucao is not None:
        verificar_salas(instancia, melhor_solucao, 1)
    gap_final = None
    if melhor_solucao is not None and math.isfinite(limite_inferior):
        gap_final = max(0.0, (melhor_objetivo - limite_inferior) /
                        max(abs(melhor_objetivo), 1e-12))

    elapsed = time.perf_counter() - inicio
    return {
        "metodo": "benders_inteiro_k1",
        "formulacao": "mestre_tipos_por_escola_subproblemas_fluxo_por_tipo",
        "status": status_final,
        "viavel": melhor_solucao is not None,
        "instancia": str(instancia_path),
        "sha256_instancia": _sha256(instancia_path),
        "k": 1,
        "unidade_distancias": instancia.unidade_distancias,
        "objetivo": (melhor_objetivo if math.isfinite(melhor_objetivo) else None),
        "limite_inferior": _finito(limite_inferior),
        "gap": gap_final,
        "iteracoes_benders": iteracoes,
        "cortes_adicionados": cortes_adicionados,
        "cortes_fracionarios_raiz": cortes_raiz,
        "rodadas_separacao_raiz": rodadas_raiz,
        "cortes_magnanti_wong_gerados": cortes_mw_gerados,
        "tempo_modelagem_s": tempo_modelagem,
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
        description="Decomposição de Benders para alocação de salas com K=1."
    )
    parser.add_argument("--instancia", type=Path, required=True)
    parser.add_argument("--time-limit", type=float, default=600.0,
                        help="limite total aproximado, em segundos")
    parser.add_argument("--mip-gap", type=float, default=0.0,
                        help="gap global de parada; 0 tenta provar ótimo")
    parser.add_argument("--master-slice", type=float, default=15.0,
                        help="tempo máximo por resolução do mestre")
    separacao_raiz = parser.add_mutually_exclusive_group()
    separacao_raiz.add_argument("--cortes-raiz", action="store_true",
                                help="separa cortes fracionários antes de cada MIP")
    separacao_raiz.add_argument("--cortes-raiz-inicial", action="store_true",
                                help="separa cortes fracionários só antes do primeiro MIP")
    parser.add_argument("--max-rodadas-raiz", type=int, default=3,
                        help="máximo de rodadas de cortes por separação na raiz")
    selecao_mw = parser.add_mutually_exclusive_group()
    selecao_mw.add_argument("--cortes-magnanti-wong", action="store_true",
                            help="escolhe todos os cortes via dual auxiliar e ponto core")
    selecao_mw.add_argument("--cortes-magnanti-wong-raiz", action="store_true",
                            help="usa Magnanti–Wong só nos cortes da raiz inicial")
    parser.add_argument("--mip-focus", type=int, choices=(0, 1, 2, 3), default=2)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--start-json", type=Path,
                        help="solução K=1 válida para iniciar o limite superior")
    parser.add_argument("--saida", type=Path,
                        default=ROOT / "resultados" / "exatos" / "candidatos" / "benders")
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
        destino = args.saida / f"{args.instancia.stem}_k1.json"
        destino.parent.mkdir(parents=True, exist_ok=True)
        temporario = destino.with_suffix(destino.suffix + ".tmp")
        temporario.write_text(json.dumps(
            result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        temporario.replace(destino)
        print(
            f"{args.instancia.name} K=1 status={result['status']} "
            f"objetivo={result['objetivo']} bound={result['limite_inferior']} "
            f"gap={result['gap']} tempo={result['tempo_total_s']:.2f}s"
        )
        print(f"Resultado salvo em {destino}")
        return 0 if result["viavel"] else 2
    except (OSError, ValueError, RuntimeError) as erro:
        print(f"Erro: {erro}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
