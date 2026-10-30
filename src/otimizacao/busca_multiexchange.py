"""Busca multi-exchange por ciclos negativos em escolas e tipos ativos.

Cada ciclo move um candidato de cada nó escola-tipo para o nó seguinte. Como
todo nó do ciclo recebe e envia um candidato do mesmo tipo, as cargas por
escola-tipo não mudam; a atribuição de salas pode ser recomposta nos quartos
que já atendiam aquele tipo. O método não altera a configuração de tipos.
"""
from dataclasses import dataclass
import math
import time
from numbers import Integral

from .salas import SolucaoSalas, avaliar_salas, verificar_salas


@dataclass(frozen=True)
class ResultadoMultiExchange:
    solucao: SolucaoSalas
    custo_inicial: float
    custo_final: float
    ciclos: int
    movimentos: int
    avaliacoes: int
    motivo_parada: str


def _nos_e_arcos(instancia, solucao, inicio, prazo):
    tipos_por_candidato = [c.tipo_prova for c in instancia.candidatos]
    membros = {}
    salas_tipo = {}
    for i, (j, r) in enumerate(solucao.alocacao):
        tipo = tipos_por_candidato[i]
        no = (j, tipo)
        membros.setdefault(no, []).append(i)
        salas_tipo.setdefault(no, set()).add(r)
    nos = sorted(membros, key=lambda item: (item[0], item[1]))
    indice = {no: pos for pos, no in enumerate(nos)}
    arcos = []
    matriz = instancia.matriz_distancias
    for no_origem in nos:
        if time.perf_counter() >= prazo:
            return nos, membros, salas_tipo, arcos, True
        j, tipo = no_origem
        origem = indice[no_origem]
        for no_destino in nos:
            k, tipo_destino = no_destino
            if j == k or tipo_destino != tipo:
                continue
            melhor = None
            for i in membros[no_origem]:
                delta = float(matriz[i][k]) - float(matriz[i][j])
                if melhor is None or (delta, i) < melhor:
                    melhor = (delta, i)
            if melhor is not None:
                arcos.append((origem, indice[no_destino], melhor[0], melhor[1]))
    return nos, membros, salas_tipo, arcos, False


def _ciclo_negativo(quantidade_nos, arcos, tolerancia):
    """Bellman-Ford com superfonte implícita; devolve arcos de um ciclo negativo."""
    if quantidade_nos < 2 or not arcos:
        return None
    dist = [0.0] * quantidade_nos
    predecessor = [None] * quantidade_nos
    atualizado = None
    for _ in range(quantidade_nos):
        atualizado = None
        for origem, destino, custo, candidato in arcos:
            proposta = dist[origem] + custo
            if proposta < dist[destino] - tolerancia:
                dist[destino] = proposta
                predecessor[destino] = (origem, candidato, custo)
                atualizado = destino
        if atualizado is None:
            return None

    no = atualizado
    for _ in range(quantidade_nos):
        anterior = predecessor[no]
        if anterior is None:
            return None
        no = anterior[0]
    inicio = no
    ciclo = []
    custo_total = 0.0
    vistos = set()
    while True:
        if no in vistos:
            if no != inicio:
                return None
            break
        vistos.add(no)
        anterior = predecessor[no]
        if anterior is None:
            return None
        origem, candidato, custo = anterior
        ciclo.append((candidato, no))
        custo_total += custo
        no = origem
    if custo_total >= -tolerancia or len({i for i, _ in ciclo}) != len(ciclo):
        return None
    return ciclo


def _reempacotar(instancia, solucao_atual, nos, salas_tipo, ciclo):
    escolas_novas = [j for j, _ in solucao_atual.alocacao]
    for candidato, destino_no in ciclo:
        escolas_novas[candidato] = nos[destino_no][0]

    por_no = {}
    for i, j in enumerate(escolas_novas):
        no = (j, instancia.candidatos[i].tipo_prova)
        por_no.setdefault(no, []).append(i)

    alocacao = [None] * len(instancia.candidatos)
    for no, candidatos in por_no.items():
        j, _ = no
        salas = sorted(salas_tipo[no])
        cursor = 0
        for r in salas:
            capacidade = instancia.escolas[j].capacidades_salas[r]
            for _ in range(capacidade):
                if cursor == len(candidatos):
                    break
                alocacao[candidatos[cursor]] = (j, r)
                cursor += 1
        if cursor != len(candidatos):
            return None
    if any(pos is None for pos in alocacao):
        return None
    return SolucaoSalas(tuple(alocacao))


def busca_multiexchange(instancia, solucao, tipos_por_local, *,
                        tempo_limite=None, max_ciclos=None,
                        tolerancia=1e-10):
    """Aplica ciclos de realocação multi-candidato que reduzem o custo.

    A busca considera, para cada par de escolas que já oferecem um mesmo tipo,
    o candidato de menor delta no nó escola-tipo de origem. Bellman-Ford
    encontra um ciclo negativo e move um candidato por nó. Cada ciclo conserva
    a quantidade de candidatos por escola-tipo, logo não excede sua capacidade
    atual nem cria novos tipos na escola. A saída é verificada por sala.
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
        raise ValueError('defina um limite de tempo ou de ciclos')
    if not math.isfinite(tolerancia) or tolerancia < 0:
        raise ValueError('tolerancia deve ser finita e não negativa')

    verificar_salas(instancia, solucao, tipos_por_local)
    inicio = time.perf_counter()
    prazo = math.inf if tempo_limite is None else inicio + tempo_limite
    atual = solucao
    custo_inicial = avaliar_salas(instancia, atual)
    custo = custo_inicial
    ciclos = movimentos = avaliacoes = 0
    motivo = 'ótimo_local_multi_exchange'

    while max_ciclos is None or ciclos < max_ciclos:
        if time.perf_counter() >= prazo:
            motivo = 'limite_tempo'
            break
        nos, _, salas_tipo, arcos, expirou = _nos_e_arcos(instancia, atual, inicio, prazo)
        avaliacoes += len(arcos)
        if expirou:
            motivo = 'limite_tempo'
            break
        ciclo = _ciclo_negativo(len(nos), arcos, tolerancia)
        if ciclo is None:
            break
        proposta = _reempacotar(instancia, atual, nos, salas_tipo, ciclo)
        if proposta is None:
            motivo = 'reparo_salas_falhou'
            break
        verificar_salas(instancia, proposta, tipos_por_local)
        novo_custo = avaliar_salas(instancia, proposta)
        if novo_custo >= custo - tolerancia:
            motivo = 'delta_nao_confirmado'
            break
        atual, custo = proposta, novo_custo
        ciclos += 1
        movimentos += len(ciclo)
    else:
        motivo = 'limite_ciclos'

    verificar_salas(instancia, atual, tipos_por_local)
    return ResultadoMultiExchange(atual, custo_inicial, custo, ciclos,
                                  movimentos, avaliacoes, motivo)
