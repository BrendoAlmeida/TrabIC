"""Enumeração exata de candidatos por sala para instâncias pequenas.

O resultado é certificado somente após esgotar a busca. Esta implementação
é independente dos construtivos e conserva aritmética exata para podas.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import isfinite, nextafter
from numbers import Integral
from time import monotonic

from src.instancias.io import validar_instancia
from src.instancias.models import Instancia
from src.otimizacao.salas import SolucaoSalas, avaliar_salas, verificar_salas


@dataclass(frozen=True)
class ResultadoExatoSalas:
    status: str
    solucao: SolucaoSalas | None
    custo: float | None
    bound: float | None
    lower_bound: float | None
    gap: float | None
    tempo_segundos: float
    combinacoes_estimadas: int | None


def _arredondar_para_baixo(valor: Fraction) -> float:
    """Converte a fração em float sem superar o valor exato."""
    convertido = float(valor)
    return nextafter(convertido, float("-inf")) if Fraction.from_float(convertido) > valor else convertido


def resolver_exato_salas(instancia: Instancia, tipos_por_local: int, *,
                         tempo_limite: float = 10.0, max_candidatos: int | None = 12,
                         max_combinacoes: int | None = 2_000_000) -> ResultadoExatoSalas:
    """Enumera atribuições a salas; limites e interrupções não certificam ótimo.

    Uma estimativa acima do teto é representada por None. ``bound`` representa
    uma cota inferior arredondada para baixo do ótimo quando certificado; ``lower_bound`` é a relaxação que
    ignora restrições, não uma cota derivada da busca interrompida.
    """
    if isinstance(tipos_por_local, bool) or not isinstance(tipos_por_local, Integral) or tipos_por_local < 1:
        raise ValueError('tipos_por_local deve ser inteiro positivo.')
    if tempo_limite is None or not isfinite(tempo_limite) or tempo_limite <= 0:
        raise ValueError('tempo_limite deve ser finito e positivo.')
    for valor, minimo, nome in ((max_candidatos, 0, 'max_candidatos'),
                                (max_combinacoes, 1, 'max_combinacoes')):
        if valor is not None and (isinstance(valor, bool) or not isinstance(valor, Integral) or valor < minimo):
            raise ValueError(f'{nome} deve ser None ou inteiro >= {minimo}.')
    validar_instancia(instancia)
    inicio = monotonic()
    n = len(instancia.candidatos)
    salas = tuple((j, r) for j, e in enumerate(instancia.escolas)
                  for r in range(len(e.capacidades_salas)))
    teto = max_combinacoes if max_combinacoes is not None else 2_000_000
    estimativa = 1
    for _ in range(n):
        if estimativa > teto // len(salas):
            estimativa = None
            break
        estimativa *= len(salas)
    if (max_candidatos is not None and n > max_candidatos) or (
            max_combinacoes is not None and estimativa is None):
        return ResultadoExatoSalas('limite', None, None, None, None, None,
                                    monotonic() - inicio, estimativa)

    distancias = tuple(tuple(Fraction.from_float(float(x)) for x in linha)
                       for linha in instancia.matriz_distancias)
    minimos = tuple(min(linha) for linha in distancias)
    sufixos = [Fraction(0)] * (n + 1)
    for i in range(n - 1, -1, -1):
        sufixos[i] = sufixos[i + 1] + minimos[i]
    cargas = [[0] * len(e.capacidades_salas) for e in instancia.escolas]
    tipo_sala = [[None] * len(e.capacidades_salas) for e in instancia.escolas]
    tipos_escola = [dict() for _ in instancia.escolas]
    alocacao = [(-1, -1)] * n
    melhor = None
    custo_exato = None
    interrompido = False

    def dfs(i: int, custo: Fraction) -> None:
        nonlocal melhor, custo_exato, interrompido
        if monotonic() - inicio >= tempo_limite:
            interrompido = True
            return
        if custo_exato is not None and custo + sufixos[i] >= custo_exato:
            return
        if i == n:
            solucao = SolucaoSalas(tuple(alocacao))
            verificar_salas(instancia, solucao, tipos_por_local)
            melhor, custo_exato = solucao, custo
            return
        tipo = instancia.candidatos[i].tipo_prova
        for j, r in salas:
            if cargas[j][r] >= instancia.escolas[j].capacidades_salas[r]:
                continue
            if tipo_sala[j][r] not in (None, tipo):
                continue
            if tipo not in tipos_escola[j] and len(tipos_escola[j]) >= tipos_por_local:
                continue
            anterior = tipo_sala[j][r]
            tipo_sala[j][r] = tipo
            cargas[j][r] += 1
            tipos_escola[j][tipo] = tipos_escola[j].get(tipo, 0) + 1
            alocacao[i] = (j, r)
            dfs(i + 1, custo + distancias[i][j])
            tipos_escola[j][tipo] -= 1
            if tipos_escola[j][tipo] == 0:
                del tipos_escola[j][tipo]
            cargas[j][r] -= 1
            tipo_sala[j][r] = anterior
            if interrompido:
                return

    dfs(0, Fraction(0))
    status = 'limite' if interrompido else ('otimo' if melhor is not None else 'inviavel')
    custo = avaliar_salas(instancia, melhor) if melhor is not None else None
    lb = _arredondar_para_baixo(sufixos[0])
    bound = _arredondar_para_baixo(custo_exato) if status == 'otimo' else None
    return ResultadoExatoSalas(status, melhor, custo, bound,
                                lb, 0.0 if status == 'otimo' else None,
                                monotonic() - inicio, estimativa)
