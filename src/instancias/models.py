"""Estruturas de dados usadas pelo importador e pelo gerador."""

from dataclasses import dataclass


TIPOS_PROVA = ("M1", "M2", "M3e", "M3s", "M3h", "M3d")


@dataclass(frozen=True)
class Candidato:
    id: str
    cep: str
    tipo_prova: str
    latitude: float
    longitude: float


@dataclass(frozen=True)
class Escola:
    id: str
    cep: str
    latitude: float
    longitude: float
    capacidades_salas: tuple[int, ...]

    @property
    def capacidade(self) -> int:
        return sum(self.capacidades_salas)


@dataclass(frozen=True)
class Instancia:
    candidatos: tuple[Candidato, ...]
    escolas: tuple[Escola, ...]
    matriz_distancias: tuple[tuple[float, ...], ...]
    unidade_distancias: str = "km"


class ErroInstancia(ValueError):
    """Indica que o arquivo ou os dados da instância são inconsistentes."""
