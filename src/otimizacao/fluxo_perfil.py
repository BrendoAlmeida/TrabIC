"""Min-cost flow exato para alocação de candidatos com perfil de salas fixo.

Cada nó escola-tipo tem capacidade igual à soma das salas atualmente ocupadas
por aquele tipo. Para cada tipo, o problema restante é uma atribuição
capacitada bipartida. O fluxo é integral; se todos os tipos terminarem, a
alocação é ótima dentro do perfil escola-tipo fornecido.
"""
from dataclasses import dataclass
import heapq
import math
import time
from numbers import Integral

from .salas import SolucaoSalas, avaliar_salas, verificar_salas


@dataclass(frozen=True)
class ResultadoFluxoPerfil:
    solucao: SolucaoSalas | None
    custo_inicial: float | None
    custo_final: float
    status: str
    tipos_otimizados: int
    aumentos_fluxo: int
    relaxacoes: int
    tempo_segundos: float


@dataclass
class _Aresta:
    destino: int
    reversa: int
    capacidade: int
    custo: float


def _adicionar(grafo, origem, destino, capacidade, custo):
    direta = _Aresta(destino, len(grafo[destino]), capacidade, float(custo))
    reversa = _Aresta(origem, len(grafo[origem]), 0, -float(custo))
    grafo[origem].append(direta)
    grafo[destino].append(reversa)
    return direta


def _resolver_tipo(instancia, candidatos, capacidades, prazo):
    """Resolve uma atribuição capacitada por sucessivos caminhos mínimos."""
    escolas = [j for j, capacidade in enumerate(capacidades) if capacidade > 0]
    n, m = len(candidatos), len(escolas)
    if not n:
        return {}, 0, 0, False
    if sum(capacidades) < n or not escolas:
        return None, 0, 0, False

    origem = 0
    no_candidato = 1
    no_escola = no_candidato + n
    destino = no_escola + m
    grafo = [[] for _ in range(destino + 1)]
    escola_indice = {j: q for q, j in enumerate(escolas)}
    referencias = [[] for _ in range(n)]
    matriz = instancia.matriz_distancias
    for q, i in enumerate(candidatos):
        if time.perf_counter() >= prazo:
            return None, 0, 0, True
        ni = no_candidato + q
        _adicionar(grafo, origem, ni, 1, 0.0)
        for j in escolas:
            custo = float(matriz[i][j])
            if not math.isfinite(custo) or custo < 0:
                raise ValueError('Custos do fluxo devem ser finitos e não negativos.')
            edge = _adicionar(grafo, ni, no_escola + escola_indice[j], 1, custo)
            referencias[q].append((j, edge))
    for j in escolas:
        _adicionar(grafo, no_escola + escola_indice[j], destino, capacidades[j], 0.0)

    potencial = [0.0] * len(grafo)
    fluxo = relaxacoes = 0
    while fluxo < n:
        dist = [math.inf] * len(grafo)
        pai_no = [-1] * len(grafo)
        pai_aresta = [-1] * len(grafo)
        dist[origem] = 0.0
        heap = [(0.0, origem)]
        while heap:
            if time.perf_counter() >= prazo:
                return None, fluxo, relaxacoes, True
            du, u = heapq.heappop(heap)
            if du > dist[u] + 1e-12:
                continue
            for q, edge in enumerate(grafo[u]):
                if edge.capacidade <= 0:
                    continue
                relaxacoes += 1
                dv = du + edge.custo + potencial[u] - potencial[edge.destino]
                if dv < dist[edge.destino] - 1e-12:
                    dist[edge.destino] = dv
                    pai_no[edge.destino] = u
                    pai_aresta[edge.destino] = q
                    heapq.heappush(heap, (dv, edge.destino))
        if not math.isfinite(dist[destino]):
            return None, fluxo, relaxacoes, False
        for v, value in enumerate(dist):
            if math.isfinite(value):
                potencial[v] += value
        v = destino
        while v != origem:
            u = pai_no[v]
            if u < 0:
                return None, fluxo, relaxacoes, False
            edge = grafo[u][pai_aresta[v]]
            edge.capacidade -= 1
            grafo[v][edge.reversa].capacidade += 1
            v = u
        fluxo += 1

    atribuicoes = {}
    for q, i in enumerate(candidatos):
        destinos = [j for j, edge in referencias[q] if edge.capacidade == 0]
        if len(destinos) != 1:
            return None, fluxo, relaxacoes, False
        atribuicoes[i] = destinos[0]
    return atribuicoes, fluxo, relaxacoes, False


def otimizar_perfil(instancia, solucao, tipos_por_local, *, tempo_limite):
    """Otimiza a atribuição para cada tipo sob o perfil de salas atual.

    As salas ocupadas no incumbente definem a capacidade de cada nó
    escola-tipo. Os tipos das salas não mudam nesta subrotina. Uma interrupção
    no meio de um tipo preserva o incumbente válido e mantém as melhorias já
    concluídas para outros tipos.
    """
    if not isinstance(solucao, SolucaoSalas):
        raise ValueError('solucao deve ser SolucaoSalas')
    if isinstance(tipos_por_local, bool) or not isinstance(tipos_por_local, Integral) or tipos_por_local < 1:
        raise ValueError('tipos_por_local deve ser inteiro positivo')
    if tempo_limite is None or not math.isfinite(tempo_limite) or tempo_limite < 0:
        raise ValueError('tempo_limite deve ser finito e não negativo')
    verificar_salas(instancia, solucao, tipos_por_local)
    inicio = time.perf_counter()
    prazo = inicio + tempo_limite
    atual = solucao
    custo_inicial = avaliar_salas(instancia, atual)
    aumentos = relaxacoes = tipos_otimizados = 0
    tipos = tuple(dict.fromkeys(c.tipo_prova for c in instancia.candidatos))
    status = 'perfil_otimizado'

    for tipo in tipos:
        if time.perf_counter() >= prazo:
            status = 'limite_tempo'
            break
        candidatos = [i for i, c in enumerate(instancia.candidatos)
                      if c.tipo_prova == tipo]
        salas_tipo = [[False] * len(e.capacidades_salas) for e in instancia.escolas]
        for i, (j, r) in enumerate(atual.alocacao):
            if instancia.candidatos[i].tipo_prova == tipo:
                salas_tipo[j][r] = True
        capacidades = [sum(e.capacidades_salas[r]
                            for r, ativa in enumerate(salas_tipo[j]) if ativa)
                       for j, e in enumerate(instancia.escolas)]
        atribuicoes, fluxo, relax, expirou = _resolver_tipo(
            instancia, candidatos, capacidades, prazo)
        aumentos += fluxo
        relaxacoes += relax
        if expirou:
            status = 'limite_tempo'
            break
        if atribuicoes is None:
            raise RuntimeError(f'Fluxo inviável para o tipo {tipo} no perfil validado.')

        alocacao = list(atual.alocacao)
        por_escola = {}
        for i, j in atribuicoes.items():
            por_escola.setdefault(j, []).append(i)
        for j, candidatos_escola in por_escola.items():
            quartos = [r for r, ativa in enumerate(salas_tipo[j]) if ativa]
            cursor = 0
            for r in quartos:
                for _ in range(instancia.escolas[j].capacidades_salas[r]):
                    if cursor == len(candidatos_escola):
                        break
                    alocacao[candidatos_escola[cursor]] = (j, r)
                    cursor += 1
            if cursor != len(candidatos_escola):
                raise RuntimeError('Falha ao empacotar a solução do fluxo nas salas do perfil.')
        proposta = SolucaoSalas(tuple(alocacao))
        verificar_salas(instancia, proposta, tipos_por_local)
        atual = proposta
        tipos_otimizados += 1

    verificar_salas(instancia, atual, tipos_por_local)
    return ResultadoFluxoPerfil(atual, custo_inicial, avaliar_salas(instancia, atual),
                                status, tipos_otimizados, aumentos, relaxacoes,
                                time.perf_counter() - inicio)


def otimizar_perfil_explicito(instancia, perfil, tipos_por_local, *, tempo_limite):
    """Resolve min-cost flow para um perfil de tipos escolhido externamente.

    ``perfil[j][r]`` é o tipo reservado ao quarto ou ``None``. Só retorna uma
    alocação quando termina o fluxo de todos os tipos; perfil insuficiente ou
    interrupção não é convertido em solução parcial.
    """
    if isinstance(tipos_por_local, bool) or not isinstance(tipos_por_local, Integral) or tipos_por_local < 1:
        raise ValueError('tipos_por_local deve ser inteiro positivo')
    if tempo_limite is None or not math.isfinite(tempo_limite) or tempo_limite < 0:
        raise ValueError('tempo_limite deve ser finito e não negativo')
    if len(perfil) != len(instancia.escolas):
        raise ValueError('perfil deve conter uma linha por escola')

    inicio = time.perf_counter()
    deadline = inicio + tempo_limite
    tipos = tuple(dict.fromkeys(c.tipo_prova for c in instancia.candidatos))
    tipos_validos = set(tipos)
    capacidades = {t: [0] * len(instancia.escolas) for t in tipos}
    for j, (escola, linha) in enumerate(zip(instancia.escolas, perfil)):
        if len(linha) != len(escola.capacidades_salas):
            raise ValueError(f'perfil tem número incorreto de quartos na escola {j}')
        oferecidos = set()
        for r, tipo in enumerate(linha):
            if tipo is None:
                continue
            if tipo not in tipos_validos:
                return ResultadoFluxoPerfil(None, None, None, 'perfil_inviavel',
                                            0, 0, 0, time.perf_counter() - inicio)
            oferecidos.add(tipo)
            capacidades[tipo][j] += escola.capacidades_salas[r]
        if len(oferecidos) > tipos_por_local:
            return ResultadoFluxoPerfil(None, None, None, 'perfil_inviavel',
                                        0, 0, 0, time.perf_counter() - inicio)

    demanda = {t: sum(c.tipo_prova == t for c in instancia.candidatos) for t in tipos}
    if any(sum(capacidades[t]) < demanda[t] for t in tipos):
        return ResultadoFluxoPerfil(None, None, None, 'perfil_inviavel',
                                    0, 0, 0, time.perf_counter() - inicio)

    alocacao = [None] * len(instancia.candidatos)
    aumentos = relaxacoes = tipos_otimizados = 0
    for tipo in tipos:
        if time.perf_counter() >= deadline:
            return ResultadoFluxoPerfil(None, None, None, 'limite_tempo',
                                        tipos_otimizados, aumentos, relaxacoes,
                                        time.perf_counter() - inicio)
        candidatos = [i for i, c in enumerate(instancia.candidatos)
                      if c.tipo_prova == tipo]
        escolas, fluxo, relax, expirou = _resolver_tipo(
            instancia, candidatos, capacidades[tipo], deadline)
        aumentos += fluxo
        relaxacoes += relax
        if expirou:
            return ResultadoFluxoPerfil(None, None, None, 'limite_tempo',
                                        tipos_otimizados, aumentos, relaxacoes,
                                        time.perf_counter() - inicio)
        if escolas is None:
            return ResultadoFluxoPerfil(None, None, None, 'perfil_inviavel',
                                        tipos_otimizados, aumentos, relaxacoes,
                                        time.perf_counter() - inicio)
        por_escola = {}
        for i, j in escolas.items():
            por_escola.setdefault(j, []).append(i)
        for j, candidatos_escola in por_escola.items():
            quartos = [r for r, reserva in enumerate(perfil[j]) if reserva == tipo]
            cursor = 0
            for r in quartos:
                for _ in range(instancia.escolas[j].capacidades_salas[r]):
                    if cursor == len(candidatos_escola):
                        break
                    alocacao[candidatos_escola[cursor]] = (j, r)
                    cursor += 1
            if cursor != len(candidatos_escola):
                return ResultadoFluxoPerfil(None, None, None, 'perfil_inviavel',
                                            tipos_otimizados, aumentos, relaxacoes,
                                            time.perf_counter() - inicio)
        tipos_otimizados += 1

    if any(pos is None for pos in alocacao):
        return ResultadoFluxoPerfil(None, None, None, 'perfil_inviavel',
                                    tipos_otimizados, aumentos, relaxacoes,
                                    time.perf_counter() - inicio)
    solucao = SolucaoSalas(tuple(alocacao))
    verificar_salas(instancia, solucao, tipos_por_local)
    custo = avaliar_salas(instancia, solucao)
    return ResultadoFluxoPerfil(solucao, None, custo, 'perfil_otimizado',
                                tipos_otimizados, aumentos, relaxacoes,
                                time.perf_counter() - inicio)
