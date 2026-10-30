"""Vizinhança de troca de salas inteiras com reconfiguração de tipos.

Dois quartos ocupados em escolas distintas trocam seus grupos de candidatos.
Cada quarto mantém um único tipo (o tipo do grupo que recebeu), e a troca só é
aceita se as capacidades e o limite K permanecerem válidos.
"""
from dataclasses import dataclass
import math
import time
from numbers import Integral

from .salas import SolucaoSalas, avaliar_salas, verificar_salas


@dataclass(frozen=True)
class ResultadoTrocaSalas:
    solucao: SolucaoSalas
    custo_inicial: float
    custo_final: float
    trocas: int
    pares_avaliados: int
    motivo_parada: str


def _estado(instancia, solucao):
    membros = [[set() for _ in e.capacidades_salas] for e in instancia.escolas]
    tipo_sala = [[None for _ in e.capacidades_salas] for e in instancia.escolas]
    tipos_escola = [dict() for _ in instancia.escolas]
    for i, (j, r) in enumerate(solucao.alocacao):
        tipo = instancia.candidatos[i].tipo_prova
        membros[j][r].add(i)
        tipo_sala[j][r] = tipo
        tipos_escola[j][tipo] = tipos_escola[j].get(tipo, 0) + 1
    return membros, tipo_sala, tipos_escola


def _tipos_apos_troca(tipos, antigo, novo):
    resultado = dict(tipos)
    resultado[antigo] -= 1
    if resultado[antigo] == 0:
        del resultado[antigo]
    resultado[novo] = resultado.get(novo, 0) + 1
    return resultado


def busca_troca_salas(instancia, solucao, tipos_por_local, *,
                      tempo_limite=None, max_trocas=None,
                      tolerancia=1e-10):
    """Aplica a melhor troca estritamente melhor entre pares de salas ocupadas.

    Ao trocar as salas, seus grupos também trocam de escola. Isso pode mudar o
    perfil de tipos de cada escola sem misturar provas dentro da sala. A busca
    termina quando esgota a vizinhança, chega ao teto configurado ou ao tempo.
    """
    if not isinstance(solucao, SolucaoSalas):
        raise ValueError('solucao deve ser SolucaoSalas')
    if isinstance(tipos_por_local, bool) or not isinstance(tipos_por_local, Integral) or tipos_por_local < 1:
        raise ValueError('tipos_por_local deve ser inteiro positivo')
    if tempo_limite is not None and (not math.isfinite(tempo_limite) or tempo_limite < 0):
        raise ValueError('tempo_limite deve ser finito e não negativo')
    if max_trocas is not None and (
            isinstance(max_trocas, bool) or not isinstance(max_trocas, Integral) or max_trocas < 0):
        raise ValueError('max_trocas deve ser None ou inteiro não negativo')
    if tempo_limite is None and max_trocas is None:
        raise ValueError('defina tempo_limite ou max_trocas')
    if not math.isfinite(tolerancia) or tolerancia < 0:
        raise ValueError('tolerancia deve ser finita e não negativa')

    verificar_salas(instancia, solucao, tipos_por_local)
    start = time.perf_counter()
    deadline = math.inf if tempo_limite is None else start + tempo_limite
    atual = solucao
    custo_inicial = avaliar_salas(instancia, atual)
    custo = custo_inicial
    trocas = pares = 0
    motivo = 'vizinhança_esgotada'
    matriz = instancia.matriz_distancias

    while max_trocas is None or trocas < max_trocas:
        if time.perf_counter() >= deadline:
            motivo = 'limite_tempo'
            break
        membros, tipo_sala, tipos_escola = _estado(instancia, atual)
        melhor = None
        melhor_delta = -tolerancia
        interromper = False
        for j, escola_j in enumerate(instancia.escolas):
            for r, capacidade_j in enumerate(escola_j.capacidades_salas):
                grupo_j = membros[j][r]
                if not grupo_j:
                    continue
                tipo_j = tipo_sala[j][r]
                for k in range(j + 1, len(instancia.escolas)):
                    for s, capacidade_k in enumerate(instancia.escolas[k].capacidades_salas):
                        if time.perf_counter() >= deadline:
                            interromper = True
                            break
                        grupo_k = membros[k][s]
                        if not grupo_k:
                            continue
                        pares += 1
                        tipo_k = tipo_sala[k][s]
                        if len(grupo_j) > capacidade_k or len(grupo_k) > capacidade_j:
                            continue
                        tipos_j = _tipos_apos_troca(tipos_escola[j], tipo_j, tipo_k)
                        tipos_k = _tipos_apos_troca(tipos_escola[k], tipo_k, tipo_j)
                        if (len(tipos_j) > tipos_por_local or
                                len(tipos_k) > tipos_por_local):
                            continue
                        delta = sum(float(matriz[i][k]) - float(matriz[i][j])
                                    for i in grupo_j)
                        delta += sum(float(matriz[i][j]) - float(matriz[i][k])
                                     for i in grupo_k)
                        if delta < melhor_delta:
                            melhor_delta = delta
                            melhor = (j, r, k, s, tuple(grupo_j), tuple(grupo_k))
                    if interromper:
                        break
                if interromper:
                    break
            if interromper:
                break
        if interromper:
            motivo = 'limite_tempo'
            break
        if melhor is None:
            break

        j, r, k, s, grupo_j, grupo_k = melhor
        nova_alocacao = list(atual.alocacao)
        for i in grupo_j:
            nova_alocacao[i] = (k, s)
        for i in grupo_k:
            nova_alocacao[i] = (j, r)
        proposta = SolucaoSalas(tuple(nova_alocacao))
        verificar_salas(instancia, proposta, tipos_por_local)
        novo_custo = avaliar_salas(instancia, proposta)
        if novo_custo >= custo - tolerancia:
            motivo = 'delta_nao_confirmado'
            break
        atual, custo = proposta, novo_custo
        trocas += 1
    else:
        motivo = 'limite_trocas'

    verificar_salas(instancia, atual, tipos_por_local)
    return ResultadoTrocaSalas(atual, custo_inicial, custo, trocas,
                               pares, motivo)
