"""Heurísticas para alocação de candidatos."""

from .modelo import avaliar, verificar
__all__ = ["avaliar", "verificar", "ResultadoHeuristica", "otimizar"]


def __getattr__(nome):
    # A força bruta e o verificador não dependem do módulo heurístico.
    if nome in ("ResultadoHeuristica", "otimizar"):
        from . import heuristica
        return getattr(heuristica, nome)
    raise AttributeError(nome)
