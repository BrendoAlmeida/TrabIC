"""Avaliação e validação independente de alocações."""

from numbers import Integral

import numpy as np


def _vetor(instancia, alocacao):
    n, m = len(instancia.candidatos), len(instancia.escolas)
    if len(alocacao) != n:
        raise ValueError(f"alocacao deve conter exatamente {n} índices.")
    for j in alocacao:
        if isinstance(j, bool) or not isinstance(j, Integral):
            raise ValueError("Cada índice de escola deve ser inteiro.")
        if j < 0 or j >= m:
            raise ValueError(f"Índice de escola fora dos limites: {j}.")
    return tuple(int(j) for j in alocacao)


def avaliar(instancia, alocacao) -> float:
    """Soma as distâncias da matriz, sem custos de abertura."""
    indices = _vetor(instancia, alocacao)
    if not indices:
        return 0.0
    matriz = np.asarray(instancia.matriz_distancias, dtype=np.float64)
    return float(matriz[np.arange(len(indices)), np.asarray(indices)].sum())


def verificar(instancia, alocacao, tipos_por_local: int) -> None:
    """Lança ValueError se a alocação não respeitar o modelo do problema."""
    if tipos_por_local not in (1, 2):
        raise ValueError("tipos_por_local deve ser 1 ou 2.")
    indices = _vetor(instancia, alocacao)
    cargas = [0] * len(instancia.escolas)
    tipos = [set() for _ in instancia.escolas]
    for i, j in enumerate(indices):
        cargas[j] += 1
        tipos[j].add(instancia.candidatos[i].tipo_prova)
    for j, (carga, usados) in enumerate(zip(cargas, tipos)):
        if carga > instancia.escolas[j].capacidade:
            raise ValueError(f"Capacidade excedida na escola {j}.")
        if len(usados) > tipos_por_local:
            raise ValueError(f"Número máximo de tipos excedido na escola {j}.")
