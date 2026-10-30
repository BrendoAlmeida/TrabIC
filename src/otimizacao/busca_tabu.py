"""Busca tabu por deslocamentos e trocas de candidatos, com salas explícitas."""
from dataclasses import dataclass
import math
import random
import time
from numbers import Integral

from .salas import SolucaoSalas, avaliar_salas, verificar_salas


@dataclass(frozen=True)
class ResultadoTabu:
    solucao: SolucaoSalas
    custo_inicial: float
    custo_final: float
    avaliacoes: int
    iteracoes: int
    melhorias: int
    motivo_parada: str


def busca_tabu(instancia, solucao, tipos_por_local, seed=0, *, tempo_limite=None,
               max_avaliacoes=1_000_000, amostra_candidatos=96,
               trocas_por_iteracao=128, tenure_min=5, tenure_max=12,
               tempo_sem_melhoria=None):
    """Explora deslocamentos viáveis e trocas entre candidatos do mesmo tipo.

    Cada iteração escolhe o melhor movimento admissível da amostra gerada.
    Um movimento inverso fica tabu por um número aleatório de iterações; a
    aspiração aceita tabu quando supera o melhor valor conhecido. O limite de
    tempo é cooperativo e soluções devolvidas são verificadas integralmente.
    """
    if not isinstance(solucao, SolucaoSalas):
        raise ValueError('solucao deve ser SolucaoSalas')
    if isinstance(tipos_por_local, bool) or not isinstance(tipos_por_local, Integral) or tipos_por_local < 1:
        raise ValueError('tipos_por_local deve ser inteiro positivo')
    if max_avaliacoes is not None and (
            isinstance(max_avaliacoes, bool) or not isinstance(max_avaliacoes, Integral) or
            max_avaliacoes < 1):
        raise ValueError('max_avaliacoes deve ser None ou inteiro >= 1')
    for name, value, low in [('amostra_candidatos', amostra_candidatos, 1),
                             ('trocas_por_iteracao', trocas_por_iteracao, 0),
                             ('tenure_min', tenure_min, 1), ('tenure_max', tenure_max, 1)]:
        if isinstance(value, bool) or not isinstance(value, Integral) or value < low:
            raise ValueError(f'{name} deve ser inteiro >= {low}')
    if tenure_max < tenure_min:
        raise ValueError('tenure_max deve ser >= tenure_min')
    if tempo_limite is not None and (not math.isfinite(tempo_limite) or tempo_limite < 0):
        raise ValueError('tempo_limite deve ser finito e não negativo')
    if tempo_sem_melhoria is not None and (
            not math.isfinite(tempo_sem_melhoria) or tempo_sem_melhoria <= 0):
        raise ValueError('tempo_sem_melhoria deve ser finito e positivo')
    if (tempo_limite is None and max_avaliacoes is None and
            tempo_sem_melhoria is None):
        raise ValueError('defina pelo menos um limite de tempo, estagnação ou avaliações')
    verificar_salas(instancia, solucao, tipos_por_local)
    start = time.perf_counter()
    deadline = None if tempo_limite is None else start + tempo_limite
    expired = lambda: deadline is not None and time.perf_counter() >= deadline
    n = len(instancia.candidatos)
    assignment = list(solucao.alocacao)
    best_assignment = assignment.copy()
    candidate_types = [c.tipo_prova for c in instancia.candidatos]
    capacities = [tuple(e.capacidades_salas) for e in instancia.escolas]
    room_members = [[set() for _ in e.capacidades_salas] for e in instancia.escolas]
    room_type = [[None for _ in e.capacidades_salas] for e in instancia.escolas]
    school_type_count = [dict() for _ in instancia.escolas]
    school_type_rooms = [dict() for _ in instancia.escolas]
    for i, (j, r) in enumerate(assignment):
        t = candidate_types[i]
        room_members[j][r].add(i)
        room_type[j][r] = t
    for j in range(len(capacities)):
        for r, members in enumerate(room_members[j]):
            if members:
                t = room_type[j][r]
                school_type_count[j][t] = school_type_count[j].get(t, 0) + 1
                school_type_rooms[j][t] = school_type_rooms[j].get(t, 0) + 1
    costs = [float(instancia.matriz_distancias[i][j]) for i, (j, _) in enumerate(assignment)]
    current_cost = sum(costs)
    best_cost = current_cost
    last_improvement = start
    rng = random.Random(seed)
    tabu = {}
    evals = iterations = improvements = 0
    reason = 'vizinhança_esgotada'

    def can_shift(i, dest):
        j, r = assignment[i]
        k, s = dest
        if dest == (j, r) or len(room_members[k][s]) >= capacities[k][s]:
            return False
        t = candidate_types[i]
        if room_type[k][s] not in (None, t):
            return False
        if k != j:
            src_type_disappears = (len(room_members[j][r]) == 1 and school_type_rooms[j].get(t, 0) == 1)
            dst_has_type = school_type_count[k].get(t, 0) > 0
            src_count = len(school_type_count[j]) - int(src_type_disappears)
            dst_count = len(school_type_count[k]) + int(not dst_has_type)
            return src_count <= tipos_por_local and dst_count <= tipos_por_local
        return True

    def apply_shift(i, dest):
        nonlocal current_cost
        j, r = assignment[i]
        k, s = dest
        t = candidate_types[i]
        room_members[j][r].remove(i)
        if not room_members[j][r]:
            room_type[j][r] = None
            school_type_count[j][t] -= 1
            school_type_rooms[j][t] -= 1
            if school_type_count[j][t] == 0:
                del school_type_count[j][t]
            if school_type_rooms[j][t] == 0:
                del school_type_rooms[j][t]
        if not room_members[k][s]:
            room_type[k][s] = t
            school_type_count[k][t] = school_type_count[k].get(t, 0) + 1
            school_type_rooms[k][t] = school_type_rooms[k].get(t, 0) + 1
        room_members[k][s].add(i)
        assignment[i] = (k, s)
        new_cost = float(instancia.matriz_distancias[i][k])
        current_cost += new_cost - costs[i]
        costs[i] = new_cost

    def apply_swap(i, q):
        nonlocal current_cost
        ji, ri = assignment[i]
        jq, rq = assignment[q]
        room_members[ji][ri].remove(i); room_members[jq][rq].remove(q)
        room_members[ji][ri].add(q); room_members[jq][rq].add(i)
        assignment[i], assignment[q] = (jq, rq), (ji, ri)
        ni = float(instancia.matriz_distancias[i][jq])
        nq = float(instancia.matriz_distancias[q][ji])
        current_cost += ni - costs[i] + nq - costs[q]
        costs[i], costs[q] = ni, nq

    def sem_melhoria():
        return (tempo_sem_melhoria is not None and
                time.perf_counter() - last_improvement >= tempo_sem_melhoria)

    def limite_avaliacoes():
        return max_avaliacoes is not None and evals >= max_avaliacoes

    while not expired() and not sem_melhoria() and not limite_avaliacoes() and n:
        iterations += 1
        order = list(range(n))
        rng.shuffle(order)
        if len(order) > amostra_candidatos:
            order = order[:amostra_candidatos]
        best_move = None
        best_move_cost = math.inf
        for i in order:
            if expired() or limite_avaliacoes():
                break
            src_j, _ = assignment[i]
            dests = []
            for k, caps in enumerate(capacities):
                if k == src_j:
                    continue
                rooms = [s for s, cap in enumerate(caps)
                         if len(room_members[k][s]) < cap and room_type[k][s] in (None, candidate_types[i])]
                if rooms:
                    s = min(rooms, key=lambda x: (caps[x] - len(room_members[k][x]), x))
                    dests.append((k, s))
            # Visit nearby schools first; retaining all schools avoids a hard cutoff.
            dests.sort(key=lambda dest: (instancia.matriz_distancias[i][dest[0]], dest))
            for dest in dests:
                if expired() or limite_avaliacoes():
                    break
                evals += 1
                if not can_shift(i, dest):
                    continue
                new_cost = current_cost - costs[i] + float(instancia.matriz_distancias[i][dest[0]])
                tabu_until = tabu.get((i, dest), -1)
                if tabu_until > iterations and new_cost >= best_cost - 1e-10:
                    continue
                if new_cost < best_move_cost:
                    best_move = ('shift', i, dest)
                    best_move_cost = new_cost
        # Same-type swaps preserve room homogeneity and school type counts, and
        # allow movement even when every compatible room is full.
        if trocas_por_iteracao:
            pairs = []
            if n > 1:
                for _ in range(trocas_por_iteracao * 3):
                    if expired() or limite_avaliacoes() or len(pairs) >= trocas_por_iteracao:
                        break
                    i, q = rng.randrange(n), rng.randrange(n)
                    if i == q or candidate_types[i] != candidate_types[q]:
                        continue
                    ji, ri = assignment[i]; jq, rq = assignment[q]
                    if ji == jq or (ji, ri) == (jq, rq):
                        continue
                    pairs.append((i, q))
            for i, q in pairs:
                if expired() or limite_avaliacoes():
                    break
                evals += 1
                ji, _ = assignment[i]; jq, _ = assignment[q]
                new_cost = current_cost - costs[i] - costs[q] + float(instancia.matriz_distancias[i][jq]) + float(instancia.matriz_distancias[q][ji])
                reverse_tabu = (tabu.get((i, (jq, rq)), -1) > iterations or
                                tabu.get((q, (ji, ri)), -1) > iterations)
                if reverse_tabu and new_cost >= best_cost - 1e-10:
                    continue
                if new_cost < best_move_cost:
                    best_move = ('swap', i, q)
                    best_move_cost = new_cost
        if best_move is None:
            reason = 'sem_movimentos_admissiveis'
            break
        if best_move[0] == 'shift':
            _, i, dest = best_move
            old = assignment[i]
            apply_shift(i, dest)
            tabu[(i, old)] = iterations + rng.randint(tenure_min, tenure_max)
        else:
            _, i, q = best_move
            old_i, old_q = assignment[i], assignment[q]
            apply_swap(i, q)
            expiry = iterations + rng.randint(tenure_min, tenure_max)
            tabu[(i, old_i)] = expiry
            tabu[(q, old_q)] = expiry
        if current_cost < best_cost - 1e-10:
            best_cost = current_cost
            best_assignment = assignment.copy()
            improvements += 1
            last_improvement = time.perf_counter()
    if expired():
        reason = 'limite_tempo'
    elif limite_avaliacoes():
        reason = 'limite_avaliacoes'
    elif sem_melhoria() and reason == 'vizinhança_esgotada':
        reason = 'estagnacao'
    result = SolucaoSalas(tuple(best_assignment))
    verificar_salas(instancia, result, tipos_por_local)
    return ResultadoTabu(result, avaliar_salas(instancia, solucao), avaliar_salas(instancia, result),
                         evals, iterations, improvements, reason)
