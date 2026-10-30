"""Busca opcional por cadeias de ejeção limitadas sobre salas.

A busca é incompleta: examina uma ordem aleatória, um número limitado de
movimentos e no máximo ``max_comprimento`` candidatos por cadeia. Estados
intermediários nunca são retornados. O tempo é consultado entre movimentos.
"""
from dataclasses import dataclass
from numbers import Integral
import math
import random
import time

from .salas import SolucaoSalas, avaliar_salas, verificar_salas


@dataclass(frozen=True)
class ResultadoCadeias:
    solucao: SolucaoSalas
    custo_inicial: float
    custo_final: float
    avaliacoes: int
    melhorias: int
    motivo_parada: str


def buscar_cadeias(instancia, solucao: SolucaoSalas, tipos_por_local: int, seed: int = 0,
                    *, max_comprimento: int = 3, max_avaliacoes: int = 10000,
                    tempo_limite: float | None = None, tolerancia: float = 1e-10) -> ResultadoCadeias:
    """Desloca candidatos encadeados e aceita apenas soluções completas melhores.

    Uma sala cheia pode receber um candidato após ejetar um ocupante. Uma sala
    parcialmente ocupada por outro tipo também pode trocar seu único ocupante.
    A busca explora no máximo um ejetado por movimento. A validade global é
    conferida ao concluir a cadeia; limites não implicam ótimo local.
    """
    verificar_salas(instancia, solucao, tipos_por_local)
    for valor, nome, minimo in ((max_comprimento, 'max_comprimento', 1),
                                 (max_avaliacoes, 'max_avaliacoes', 0)):
        if isinstance(valor, bool) or not isinstance(valor, Integral) or valor < minimo:
            raise ValueError(f'{nome} deve ser inteiro >= {minimo}.')
    if tempo_limite is not None and (not math.isfinite(tempo_limite) or tempo_limite < 0):
        raise ValueError('tempo_limite deve ser finito e não negativo.')
    if not math.isfinite(tolerancia) or tolerancia < 0:
        raise ValueError('tolerancia deve ser finita e não negativa.')
    rng = random.Random(seed)
    inicio = time.monotonic()
    custo_inicial = avaliar_salas(instancia, solucao)
    custo = custo_inicial
    atual = solucao
    avaliacoes = melhorias = 0
    motivo = 'cadeias_examinadas'
    salas = [(j, r) for j, e in enumerate(instancia.escolas)
             for r in range(len(e.capacidades_salas)) if e.capacidades_salas[r] > 0]

    def limite():
        nonlocal motivo
        if tempo_limite is not None and time.monotonic() - inicio >= tempo_limite:
            motivo = 'limite_tempo'
            return True
        if avaliacoes >= max_avaliacoes:
            motivo = 'limite_avaliacoes'
            return True
        return False

    while not limite():
        atribuicoes = list(atual.alocacao)
        membros = {s: [] for s in salas}
        for i, sala in enumerate(atribuicoes):
            membros[sala].append(i)
        candidatos = list(range(len(atribuicoes)))
        rng.shuffle(candidatos)
        encontrada = None

        def visitar(i, visitados, origem):
            nonlocal avaliacoes, encontrada
            opcoes = list(salas)
            rng.shuffle(opcoes)
            tipo = instancia.candidatos[i].tipo_prova
            for sala in opcoes:
                if limite():
                    return
                if sala == origem:
                    continue
                j, r = sala
                presentes = membros[sala]
                capacidade = instancia.escolas[j].capacidades_salas[r]
                opcoes_ejecao = []
                if len(presentes) < capacidade and (not presentes or instancia.candidatos[presentes[0]].tipo_prova == tipo):
                    opcoes_ejecao.append(None)
                if len(visitados) < max_comprimento:
                    opcoes_ejecao.extend(k for k in presentes if k not in visitados and
                                          all(instancia.candidatos[x].tipo_prova == tipo for x in presentes if x != k))
                rng.shuffle(opcoes_ejecao)
                for ejetado in opcoes_ejecao:
                    if limite():
                        return
                    avaliacoes += 1
                    if ejetado is not None:
                        presentes.remove(ejetado)
                    presentes.append(i)
                    atribuicoes[i] = sala
                    if ejetado is None:
                        proposta = SolucaoSalas(tuple(atribuicoes))
                        try:
                            verificar_salas(instancia, proposta, tipos_por_local)
                        except ValueError:
                            pass
                        else:
                            if avaliar_salas(instancia, proposta) < custo - tolerancia:
                                encontrada = proposta
                    else:
                        visitar(ejetado, visitados | {ejetado}, sala)
                    presentes.remove(i)
                    if ejetado is not None:
                        presentes.append(ejetado)
                        atribuicoes[ejetado] = sala
                    if encontrada is not None or limite():
                        return

        for i in candidatos:
            if limite():
                break
            origem = atribuicoes[i]
            membros[origem].remove(i)
            visitar(i, {i}, origem)
            membros[origem].append(i)
            atribuicoes[i] = origem
            if encontrada is not None:
                break
        if encontrada is None:
            break
        atual = encontrada
        custo = avaliar_salas(instancia, atual)
        melhorias += 1
    verificar_salas(instancia, atual, tipos_por_local)
    return ResultadoCadeias(atual, custo_inicial, custo, avaliacoes, melhorias, motivo)
