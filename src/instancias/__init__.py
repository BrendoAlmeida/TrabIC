"""Importação e geração de instâncias para alocação de candidatos."""

from .generator import gerar_instancias
from .io import carregar_instancia, ler_instancia, salvar_instancia
from .models import Candidato, Escola, Instancia

__all__ = [
    "Candidato",
    "Escola",
    "Instancia",
    "carregar_instancia",
    "ler_instancia",
    "gerar_instancias",
    "salvar_instancia",
]
