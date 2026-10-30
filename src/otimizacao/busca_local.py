"""Busca local VND sobre alocações por sala.

As vizinhanças são limitadas: parada por orçamento não implica ótimo local.
"""
from dataclasses import dataclass
from numbers import Integral
import math
from itertools import product
import random
import time

from .salas import SolucaoSalas, avaliar_salas, verificar_salas


@dataclass(frozen=True)
class ResultadoBusca:
    solucao: SolucaoSalas
    custo_inicial: float
    custo_final: float
    avaliacoes: int
    melhorias: int
    motivo_parada: str


def delta_shift(instancia, candidato, origem, destino):
    """Variação do custo pela mudança de escola de um candidato."""
    return (instancia.matriz_distancias[candidato][destino]
            - instancia.matriz_distancias[candidato][origem])


def delta_swap(instancia, primeiro, segundo, escola_primeiro, escola_segundo):
    """Variação do custo pela troca de escola de dois candidatos."""
    return (delta_shift(instancia, primeiro, escola_primeiro, escola_segundo)
            + delta_shift(instancia, segundo, escola_segundo, escola_primeiro))


def vnd(instancia, solucao, tipos_por_local, seed=0, *, max_avaliacoes=100000,
        tempo_limite=None, tolerancia=1e-10,
        vizinhancas=("shift", "swap", "reconfiguracao")):
    """First improvement: shift, swap e reconstrução de duas salas.

    A reconfiguração libera duas salas e reconstrói seus tipos e ocupantes. A
    busca avalia no máximo `max_avaliacoes` movimentos, e um limite de tempo
    opcional também é verificado entre movimentos. O orçamento de avaliações é
    dividido igualmente entre vizinhanças; pode limitar o reinício completo. A
    ordem usa a seed. `vizinhancas` escolhe a ordem e o subconjunto de
    "shift", "swap" e "reconfiguracao". A reconstrução é heurística e restrita aos candidatos
    das duas salas escolhidas; não explora configurações de mais salas.
    """
    verificar_salas(instancia, solucao, tipos_por_local)
    if isinstance(max_avaliacoes, bool) or not isinstance(max_avaliacoes, Integral) or max_avaliacoes < 0:
        raise ValueError('max_avaliacoes deve ser inteiro não negativo')
    if tempo_limite is not None and (not math.isfinite(tempo_limite) or tempo_limite < 0):
        raise ValueError('tempo_limite deve ser não negativo e finito')
    if not math.isfinite(tolerancia) or tolerancia < 0:
        raise ValueError('tolerancia deve ser não negativa e finita')
    if not isinstance(vizinhancas, (tuple, list)) or not vizinhancas:
        raise ValueError('vizinhancas deve ser uma sequência não vazia')
    vizinhancas = tuple(vizinhancas)
    if (any(v not in ('shift', 'swap', 'reconfiguracao') for v in vizinhancas)
            or len(set(vizinhancas)) != len(vizinhancas)):
        raise ValueError('vizinhanças inválidas ou repetidas')
    rng = random.Random(seed)
    inicio = time.perf_counter()
    aloc = list(solucao.alocacao)
    n = len(aloc)
    posicoes = [(j, r) for j, e in enumerate(instancia.escolas)
                for r in range(len(e.capacidades_salas))]
    custo_inicial = avaliar_salas(instancia, solucao)
    avaliacoes = melhorias = 0
    motivo = 'vizinhanças_examinadas'

    def esgotado():
        nonlocal motivo
        if tempo_limite is not None and time.perf_counter() - inicio >= tempo_limite:
            motivo = 'tempo_limite'
            return True
        if avaliacoes >= max_avaliacoes:
            motivo = 'limite_avaliacoes'
            return True
        return False

    def estado():
        membros = {p: [] for p in posicoes}
        for i, p in enumerate(aloc):
            membros[p].append(i)
        return membros

    def aceitar(mudancas, delta):
        nonlocal melhorias
        if delta >= -tolerancia:
            return False
        candidato = aloc.copy()
        for i, p in mudancas:
            candidato[i] = p
        nova = SolucaoSalas(tuple(candidato))
        try:
            verificar_salas(instancia, nova, tipos_por_local)
        except ValueError:
            return False
        # Reavaliação integral evita aceitar melhora apenas por arredondamento.
        antiga = SolucaoSalas(tuple(aloc))
        if avaliar_salas(instancia, nova) >= avaliar_salas(instancia, antiga) - tolerancia:
            return False
        aloc[:] = candidato
        melhorias += 1
        return True

    vizinhanca = 0
    qtd_vizinhancas = len(vizinhancas)
    cotas = [max_avaliacoes // qtd_vizinhancas + (i < max_avaliacoes % qtd_vizinhancas)
             for i in range(qtd_vizinhancas)]
    contagens = [0] * qtd_vizinhancas
    while vizinhanca < qtd_vizinhancas and not esgotado():
        if contagens[vizinhanca] >= cotas[vizinhanca]:
            vizinhanca += 1
            continue
        membros = estado()
        ordem_candidatos = list(range(n))
        rng.shuffle(ordem_candidatos)
        ordem_salas = posicoes.copy()
        rng.shuffle(ordem_salas)
        melhorou = False
        def fim_vizinhanca():
            return esgotado() or contagens[vizinhanca] >= cotas[vizinhanca]

        def contar():
            nonlocal avaliacoes
            avaliacoes += 1
            contagens[vizinhanca] += 1

        if vizinhancas[vizinhanca] == 'shift':
            for i in ordem_candidatos:
                origem = aloc[i]
                for destino in ordem_salas:
                    if fim_vizinhanca():
                        break
                    contar()
                    if origem == destino:
                        continue
                    delta = delta_shift(instancia, i, origem[0], destino[0])
                    if aceitar([(i, destino)], delta):
                        melhorou = True
                        break
                if melhorou or fim_vizinhanca():
                    break
        elif vizinhancas[vizinhanca] == 'swap':
            # Examinar pares em ordem aleatória reprodutível, sem materializar N².
            for a, i in enumerate(ordem_candidatos):
                for k in ordem_candidatos[a + 1:]:
                    if fim_vizinhanca():
                        break
                    contar()
                    if aloc[i] == aloc[k] or aloc[i][0] == aloc[k][0]:
                        continue
                    delta = delta_swap(instancia, i, k, aloc[i][0], aloc[k][0])
                    if aceitar([(i, aloc[k]), (k, aloc[i])], delta):
                        melhorou = True
                        break
                if melhorou or fim_vizinhanca():
                    break
        else:
            # Libera duas salas e tenta configurações dos tipos presentes. Para
            # um tipo em ambas, a ordenação por delta determina a melhor divisão.
            for a, p in enumerate(ordem_salas):
                for q in ordem_salas[a + 1:]:
                    if fim_vizinhanca():
                        break
                    if p[0] == q[0] or not (membros[p] or membros[q]):
                        contar()
                        continue
                    grupo = membros[p] + membros[q]
                    tipos = tuple(dict.fromkeys(instancia.candidatos[i].tipo_prova for i in grupo))
                    por_tipo = {t: [i for i in grupo if instancia.candidatos[i].tipo_prova == t]
                                for t in tipos}
                    for tipo_p, tipo_q in product(tipos, repeat=2):
                        if fim_vizinhanca():
                            break
                        contar()
                        if any(t not in (tipo_p, tipo_q) for t in tipos):
                            continue
                        cp = instancia.escolas[p[0]].capacidades_salas[p[1]]
                        cq = instancia.escolas[q[0]].capacidades_salas[q[1]]
                        if tipo_p == tipo_q:
                            if len(grupo) > cp + cq:
                                continue
                            ordenados = sorted(grupo, key=lambda i: delta_shift(instancia, i, q[0], p[0]))
                            qtd_p = min(cp, len(grupo))
                            # Usar até cp vagas quando p é mais barata; garantir
                            # que o restante caiba em q.
                            qtd_p = max(len(grupo) - cq, sum(
                                delta_shift(instancia, i, q[0], p[0]) < 0 for i in ordenados[:qtd_p]))
                            para_p, para_q = ordenados[:qtd_p], ordenados[qtd_p:]
                        else:
                            para_p, para_q = por_tipo[tipo_p], por_tipo[tipo_q]
                            if len(para_p) > cp or len(para_q) > cq:
                                continue
                        alteracoes = [(i, p) for i in para_p] + [(i, q) for i in para_q]
                        delta = sum(delta_shift(instancia, i, aloc[i][0], destino[0])
                                    for i, destino in alteracoes)
                        if aceitar(alteracoes, delta):
                            melhorou = True
                            break
                    if melhorou or fim_vizinhanca():
                        break
                if melhorou or fim_vizinhanca():
                    break
        vizinhanca = 0 if melhorou else vizinhanca + 1
    if motivo == 'vizinhanças_examinadas' and any(
            usados >= c for usados, c in zip(contagens, cotas)):
        motivo = 'limite_vizinhanca'
    final = SolucaoSalas(tuple(aloc))
    verificar_salas(instancia, final, tipos_por_local)
    return ResultadoBusca(final, custo_inicial, avaliar_salas(instancia, final),
                          avaliacoes, melhorias, motivo)
