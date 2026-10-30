"""Vizinhança de ciclos de grupos de candidatos entre salas.

Um ciclo move todo o grupo de cada quarto ocupado ao próximo quarto do ciclo.
Isso reconfigura tipos de sala e escolas em bloco. Arcos individuais já
respeitam capacidade e K após remover o tipo do quarto que sai e inserir o
tipo do grupo que entra; pares e triplas usam no máximo um quarto por escola.
"""
from dataclasses import dataclass
from itertools import combinations
import math
import time
from numbers import Integral

from .salas import SolucaoSalas, avaliar_salas, verificar_salas


@dataclass(frozen=True)
class ResultadoCiclosSalas:
    solucao: SolucaoSalas
    custo_inicial: float
    custo_final: float
    ciclos: int
    quartos_movidos: int
    candidatos_movidos: int
    avaliacoes: int
    motivo_parada: str


def _estado(instancia, solucao):
    membros = [[set() for _ in e.capacidades_salas] for e in instancia.escolas]
    tipos = [[None for _ in e.capacidades_salas] for e in instancia.escolas]
    contagens = [dict() for _ in instancia.escolas]
    for i, (j, r) in enumerate(solucao.alocacao):
        tipo = instancia.candidatos[i].tipo_prova
        membros[j][r].add(i)
        tipos[j][r] = tipo
        contagens[j][tipo] = contagens[j].get(tipo, 0) + 1
    quartos = [(j, r) for j, e in enumerate(instancia.escolas)
               for r in range(len(e.capacidades_salas)) if membros[j][r]]
    return quartos, membros, tipos, contagens


def _grafo(instancia, quartos, membros, tipos, contagens, tipos_por_local,
           deadline):
    n = len(quartos)
    permitido = [[False] * n for _ in range(n)]
    delta = [[math.inf] * n for _ in range(n)]
    salas = [[None] * n for _ in range(n)]
    pares = 0
    matriz = instancia.matriz_distancias
    for u, (j, r) in enumerate(quartos):
        if time.perf_counter() >= deadline:
            return permitido, delta, salas, pares, True
        grupo = membros[j][r]
        tipo_origem = tipos[j][r]
        for v, (k, s) in enumerate(quartos):
            if j == k:
                continue
            pares += 1
            if len(grupo) > instancia.escolas[k].capacidades_salas[s]:
                continue
            tipo_removido = tipos[k][s]
            tipos_novos = set(contagens[k])
            if contagens[k][tipo_removido] == 1:
                tipos_novos.remove(tipo_removido)
            tipos_novos.add(tipo_origem)
            if len(tipos_novos) > tipos_por_local:
                continue
            permitido[u][v] = True
            salas[u][v] = (k, s)
            delta[u][v] = sum(float(matriz[i][k]) - float(matriz[i][j])
                              for i in grupo)
    return permitido, delta, salas, pares, False


def _melhor_ciclo(quartos, permitido, delta, deadline, tolerancia):
    melhor_delta = -tolerancia
    melhor = None
    avaliacoes = 0
    n = len(quartos)
    # Ciclos de dois quartos.
    for u in range(n):
        j, _ = quartos[u]
        for v in range(u + 1, n):
            if time.perf_counter() >= deadline:
                return melhor, avaliacoes, True
            if quartos[v][0] == j or not permitido[u][v] or not permitido[v][u]:
                continue
            avaliacoes += 1
            score = delta[u][v] + delta[v][u]
            if score < melhor_delta:
                melhor_delta = score
                melhor = (u, v)

    # Ciclos de três quartos, sempre em escolas distintas.
    for a, b, c in combinations(range(n), 3):
        if time.perf_counter() >= deadline:
            return melhor, avaliacoes, True
        ja, jb, jc = quartos[a][0], quartos[b][0], quartos[c][0]
        if ja == jb or ja == jc or jb == jc:
            continue
        for ciclo in ((a, b, c), (a, c, b)):
            u, v, w = ciclo
            if permitido[u][v] and permitido[v][w] and permitido[w][u]:
                avaliacoes += 1
                score = delta[u][v] + delta[v][w] + delta[w][u]
                if score < melhor_delta:
                    melhor_delta = score
                    melhor = ciclo
    return melhor, avaliacoes, False


def busca_ciclos_salas(instancia, solucao, tipos_por_local, *,
                       tempo_limite=None, max_ciclos=None,
                       tolerancia=1e-10):
    """Aplica ciclos negativos de dois ou três quartos ocupados.

    O movimento troca simultaneamente os grupos completos de candidatos nos
    quartos do ciclo. O tipo do quarto acompanha seu novo grupo. Cada escola
    aparece no máximo uma vez por ciclo, e os tipos/capacidades são verificados
    antes de aceitar a solução. O movimento pode reconfigurar tipos, mas não
    divide os grupos de candidatos presentes nos quartos selecionados.
    """
    if not isinstance(solucao, SolucaoSalas):
        raise ValueError('solucao deve ser SolucaoSalas')
    if isinstance(tipos_por_local, bool) or not isinstance(tipos_por_local, Integral) or tipos_por_local < 1:
        raise ValueError('tipos_por_local deve ser inteiro positivo')
    if tempo_limite is not None and (not math.isfinite(tempo_limite) or tempo_limite < 0):
        raise ValueError('tempo_limite deve ser finito e não negativo')
    if max_ciclos is not None and (
            isinstance(max_ciclos, bool) or not isinstance(max_ciclos, Integral) or max_ciclos < 0):
        raise ValueError('max_ciclos deve ser None ou inteiro não negativo')
    if tempo_limite is None and max_ciclos is None:
        raise ValueError('defina tempo_limite ou max_ciclos')
    if not math.isfinite(tolerancia) or tolerancia < 0:
        raise ValueError('tolerancia deve ser finita e não negativa')

    verificar_salas(instancia, solucao, tipos_por_local)
    inicio = time.perf_counter()
    deadline = math.inf if tempo_limite is None else inicio + tempo_limite
    atual = solucao
    custo_inicial = avaliar_salas(instancia, atual)
    custo = custo_inicial
    ciclos = quartos_movidos = candidatos_movidos = avaliacoes = 0
    motivo = 'ótimo_local_ciclos_salas'

    while max_ciclos is None or ciclos < max_ciclos:
        if time.perf_counter() >= deadline:
            motivo = 'limite_tempo'
            break
        quartos, membros, tipos, contagens = _estado(instancia, atual)
        permitido, delta, salas, pares, expirou = _grafo(
            instancia, quartos, membros, tipos, contagens, tipos_por_local, deadline)
        if expirou:
            motivo = 'limite_tempo'
            break
        ciclo, examinados, expirou = _melhor_ciclo(
            quartos, permitido, delta, deadline, tolerancia)
        avaliacoes += pares + examinados
        if expirou:
            motivo = 'limite_tempo'
            break
        if ciclo is None:
            break

        alocacao = list(atual.alocacao)
        movimentos = []
        for pos, origem in enumerate(ciclo):
            destino = ciclo[(pos + 1) % len(ciclo)]
            grupo = tuple(membros[quartos[origem][0]][quartos[origem][1]])
            movimentos.append((grupo, salas[origem][destino]))
        for grupo, destino in movimentos:
            for i in grupo:
                alocacao[i] = destino
        proposta = SolucaoSalas(tuple(alocacao))
        verificar_salas(instancia, proposta, tipos_por_local)
        novo_custo = avaliar_salas(instancia, proposta)
        if novo_custo >= custo - tolerancia:
            motivo = 'delta_nao_confirmado'
            break
        atual, custo = proposta, novo_custo
        ciclos += 1
        quartos_movidos += len(ciclo)
        candidatos_movidos += sum(len(grupo) for grupo, _ in movimentos)
    else:
        motivo = 'limite_ciclos'

    verificar_salas(instancia, atual, tipos_por_local)
    return ResultadoCiclosSalas(atual, custo_inicial, custo, ciclos,
                                quartos_movidos, candidatos_movidos,
                                avaliacoes, motivo)
