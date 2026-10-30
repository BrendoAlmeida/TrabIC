"""Representação e verificação independente de alocações por sala.

As posições são índices de base zero na ordem original da instância. Os tipos e
ocupações são derivados da alocação, e não de caches da heurística.
"""

from dataclasses import dataclass
from numbers import Integral


@dataclass(frozen=True)
class SolucaoSalas:
    """Uma posição (escola, sala) por candidato, na ordem da entrada."""

    alocacao: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class SalaResumo:
    indice_escola: int
    id_escola: str
    indice_sala: int
    capacidade: int
    tipo: str | None
    indices_candidatos: tuple[int, ...]
    ids_candidatos: tuple[str, ...]


def _indices(instancia, solucao: SolucaoSalas) -> tuple[tuple[int, int], ...]:
    if not isinstance(solucao, SolucaoSalas):
        raise ValueError("A solução deve ser uma SolucaoSalas.")
    if len(solucao.alocacao) != len(instancia.candidatos):
        raise ValueError("A alocação deve conter uma posição para cada candidato.")
    resultado = []
    for posicao in solucao.alocacao:
        if not isinstance(posicao, (tuple, list)) or len(posicao) != 2:
            raise ValueError("Cada posição deve conter índices de escola e sala.")
        j, r = posicao
        if any(isinstance(v, bool) or not isinstance(v, Integral) for v in (j, r)):
            raise ValueError("Índices de escola e sala devem ser inteiros.")
        if j < 0 or j >= len(instancia.escolas):
            raise ValueError(f"Índice de escola fora dos limites: {j}.")
        if r < 0 or r >= len(instancia.escolas[j].capacidades_salas):
            raise ValueError(f"Índice de sala fora dos limites: escola {j}, sala {r}.")
        resultado.append((int(j), int(r)))
    return tuple(resultado)


def verificar_salas(instancia, solucao: SolucaoSalas, tipos_por_local: int) -> None:
    """Lança ValueError se a solução violar atribuição, sala ou limite de tipos."""
    if (isinstance(tipos_por_local, bool) or
            not isinstance(tipos_por_local, Integral) or tipos_por_local < 1):
        raise ValueError("tipos_por_local deve ser um inteiro positivo.")
    indices = _indices(instancia, solucao)
    ocupacoes = [[0] * len(e.capacidades_salas) for e in instancia.escolas]
    tipos_salas = [[None] * len(e.capacidades_salas) for e in instancia.escolas]
    tipos_escolas = [set() for _ in instancia.escolas]
    for i, (j, r) in enumerate(indices):
        tipo = instancia.candidatos[i].tipo_prova
        anterior = tipos_salas[j][r]
        if anterior is not None and anterior != tipo:
            raise ValueError(f"Tipos diferentes na escola {j}, sala {r}.")
        tipos_salas[j][r] = tipo
        ocupacoes[j][r] += 1
        if ocupacoes[j][r] > instancia.escolas[j].capacidades_salas[r]:
            raise ValueError(f"Capacidade excedida na escola {j}, sala {r}.")
        tipos_escolas[j].add(tipo)
        if len(tipos_escolas[j]) > tipos_por_local:
            raise ValueError(f"Número máximo de tipos excedido na escola {j}.")


def avaliar_salas(instancia, solucao: SolucaoSalas) -> float:
    """Soma custos por candidato e escola; não certifica viabilidade."""
    indices = _indices(instancia, solucao)
    return float(sum(instancia.matriz_distancias[i][j] for i, (j, _) in enumerate(indices)))


def resumo_salas(instancia, solucao: SolucaoSalas, tipos_por_local: int) -> tuple[SalaResumo, ...]:
    """Retorna todas as salas em ordem de entrada; vazias têm tipo None."""
    verificar_salas(instancia, solucao, tipos_por_local)
    membros = [[[] for _ in e.capacidades_salas] for e in instancia.escolas]
    for i, (j, r) in enumerate(solucao.alocacao):
        membros[j][r].append(i)
    resultado = []
    for j, escola in enumerate(instancia.escolas):
        for r, capacidade in enumerate(escola.capacidades_salas):
            ids = tuple(membros[j][r])
            tipo = instancia.candidatos[ids[0]].tipo_prova if ids else None
            resultado.append(SalaResumo(j, escola.id, r, capacidade, tipo, ids,
                                        tuple(instancia.candidatos[i].id for i in ids)))
    return tuple(resultado)
