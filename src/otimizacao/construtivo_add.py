"""Construtivo alternativo por lotes sala/tipo, com reinícios limitados.

A seleção usa distância média e protege capacidade residual com uma condição
necessária, não suficiente. A recuperação por configuração é heurística e a
falha de construção nunca é uma prova de inviabilidade.
"""
from numbers import Integral
import random
import math
import time

import numpy as np

from .construtivo import ResultadoConstrucao, _recuperar_configuracao
from .salas import SolucaoSalas, verificar_salas


def construir_add(instancia, tipos_por_local: int, seed: int = 0, *, alpha: float = 0.2,
                  reinicios: int = 10, limite_reparacao: int = 100,
                  limite_configuracoes: int = 30000, prazo: float | None = None) -> ResultadoConstrucao:
    """Seleciona sala livre e tipo, alocando os mais próximos em lote.

    A RCL contém custos médios <= mínimo + alpha*(máximo-mínimo). Antes de
    selecionar, rejeita lotes que deixem algum tipo sem capacidade potencial
    suficiente, quando há alternativas que passam nesse filtro. O parâmetro
    limite_reparacao controla o número de tentativas de remontagem completa a
    partir de uma configuração, limitado a três; cada uma usa até
    limite_configuracoes passos.
    """
    if isinstance(tipos_por_local, bool) or not isinstance(tipos_por_local, Integral) or tipos_por_local < 1:
        raise ValueError('tipos_por_local deve ser um inteiro positivo')
    if not 0 <= alpha <= 1:
        raise ValueError('alpha deve estar entre 0 e 1')
    for nome, valor, minimo in [('reinicios', reinicios, 1), ('limite_reparacao', limite_reparacao, 0), ('limite_configuracoes', limite_configuracoes, 0)]:
        if isinstance(valor, bool) or not isinstance(valor, Integral) or valor < minimo:
            raise ValueError(f'{nome} deve ser um inteiro >= {minimo}')
    if prazo is not None and not math.isfinite(prazo):
        raise ValueError('prazo deve ser finito')
    expirou = lambda: prazo is not None and time.perf_counter() >= prazo
    if expirou():
        return ResultadoConstrucao('tempo_limite', None, 0, 0, 'prazo global esgotado')
    n, m = len(instancia.candidatos), len(instancia.escolas)
    if n == 0:
        solucao = SolucaoSalas(())
        verificar_salas(instancia, solucao, tipos_por_local)
        return ResultadoConstrucao('viavel', solucao, 1, 0, '')
    if not m:
        return ResultadoConstrucao('falha_heuristica', None, 0, 0, 'nenhuma escola disponível')
    matriz = np.asarray(instancia.matriz_distancias, dtype=float)
    if matriz.shape != (n, m) or not np.isfinite(matriz).all():
        raise ValueError('Matriz de distâncias inválida')
    tipos = tuple(dict.fromkeys(c.tipo_prova for c in instancia.candidatos))
    codigos = np.array([tipos.index(c.tipo_prova) for c in instancia.candidatos])
    caps = [tuple(e.capacidades_salas) for e in instancia.escolas]
    rng = random.Random(seed)
    # Ordenação é feita uma vez, preservando a ordem de entrada nos empates.
    ordenados = [[np.flatnonzero(codigos == t)[np.argsort(matriz[codigos == t, j], kind='stable')]
                  for t in range(len(tipos))] for j in range(m)]
    for tentativa in range(1, reinicios + 1):
        if expirou():
            return ResultadoConstrucao('tempo_limite', None, tentativa - 1, 0, 'prazo global esgotado')
        pendentes = np.ones(n, dtype=bool)
        demanda = [int(np.sum(codigos == t)) for t in range(len(tipos))]
        al = [None] * n
        livres = [[q > 0 for q in escola] for escola in caps]
        usados = [set() for _ in caps]
        while any(demanda):
            if expirou():
                return ResultadoConstrucao('tempo_limite', None, tentativa, 0, 'prazo global esgotado')
            opcoes = []
            for j, salas in enumerate(caps):
                if expirou():
                    return ResultadoConstrucao('tempo_limite', None, tentativa, 0, 'prazo global esgotado')
                for r, cap in enumerate(salas):
                    if not livres[j][r]:
                        continue
                    for t in range(len(tipos)):
                        if not demanda[t] or (t not in usados[j] and len(usados[j]) >= tipos_por_local):
                            continue
                        candidatos = ordenados[j][t]
                        candidatos = candidatos[pendentes[candidatos]][:cap]
                        if not len(candidatos):
                            continue
                        # Capacidade potencial é superestimada: os tipos ainda
                        # disputam salas. Portanto, não certifica viabilidade.
                        restante = demanda.copy()
                        restante[t] -= len(candidatos)
                        possivel = True
                        for u, precisos in enumerate(restante):
                            if not precisos:
                                continue
                            total = 0
                            for jj, ss in enumerate(caps):
                                tipos_j = usados[jj] | ({t} if jj == j else set())
                                if u not in tipos_j and len(tipos_j) >= tipos_por_local:
                                    continue
                                total += sum(c for rr, c in enumerate(ss)
                                             if livres[jj][rr] and (jj, rr) != (j, r))
                            if total < precisos:
                                possivel = False
                                break
                        opcoes.append((float(np.mean(matriz[candidatos, j])), j, r, t, candidatos, possivel))
            if not opcoes:
                break
            filtradas = [op for op in opcoes if op[5]]
            # Um filtro necessário que rejeita todas as opções não deve criar
            # falsa certificação de inviabilidade; reinício/recuperação seguem.
            if not filtradas:
                break
            minimo = min(op[0] for op in filtradas)
            maximo = max(op[0] for op in filtradas)
            rcl = [op for op in filtradas if op[0] <= minimo + alpha * (maximo - minimo)]
            _, j, r, t, candidatos, _ = rng.choice(rcl)
            for i in candidatos:
                al[int(i)] = (j, r)
            pendentes[candidatos] = False
            demanda[t] -= len(candidatos)
            livres[j][r] = False
            usados[j].add(t)
        if not any(demanda):
            solucao = SolucaoSalas(tuple(al))
            verificar_salas(instancia, solucao, tipos_por_local)
            return ResultadoConstrucao('viavel', solucao, tentativa, 0, '')
    # Reparação limitada por remontagem global, sem reaproveitar estado parcial.
    # Até três remontagens evitam que o default gere centenas de buscas longas.
    for reparacoes in range(1, min(limite_reparacao, 3) + 1):
        if expirou():
            return ResultadoConstrucao('tempo_limite', None, reinicios, reparacoes - 1, 'prazo global esgotado')
        solucao = _recuperar_configuracao(instancia, tipos_por_local, rng, limite_configuracoes, prazo=prazo)
        if solucao is not None:
            verificar_salas(instancia, solucao, tipos_por_local)
            return ResultadoConstrucao('viavel', solucao, reinicios, reparacoes, '')
    if expirou():
        return ResultadoConstrucao('tempo_limite', None, reinicios, min(limite_reparacao, 3), 'prazo global esgotado')
    return ResultadoConstrucao('falha_heuristica', None, reinicios,
                              min(limite_reparacao, 3) if limite_configuracoes else 0,
                              'tentativas de construção e recuperação esgotadas')
