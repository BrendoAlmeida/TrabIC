"""Modelo legado agregado por escola, sem restrições de salas.

Não certifica soluções do problema por sala; use exato_salas para esse fim.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import isfinite
from time import monotonic

from src.instancias.models import Instancia
from src.instancias.io import validar_instancia
from src.otimizacao.modelo import avaliar, verificar


MAX_CANDIDATOS_PADRAO = 12
MAX_COMBINACOES_PADRAO = 2_000_000
TEMPO_LIMITE_PADRAO = 10.0


@dataclass(frozen=True)
class ResultadoExato:
    status: str
    alocacao: tuple[int, ...] | None
    custo: float | None
    bound: float | None
    lower_bound: float | None
    gap: float | None
    tempo_segundos: float
    combinacoes_estimadas: int | None


def resolver_exato(
    instancia: Instancia,
    tipos_por_local: int,
    *,
    tempo_limite: float = TEMPO_LIMITE_PADRAO,
    max_candidatos: int | None = MAX_CANDIDATOS_PADRAO,
    max_combinacoes: int | None = MAX_COMBINACOES_PADRAO,
) -> ResultadoExato:
    """Enumera alocações; retorna ``limite`` se não puder certificar ótimo.

    Quando ótimo, ``bound`` é o custo certificado e ``lower_bound`` conserva
    a relaxação independente. Em limite, ``bound`` é None e ``lower_bound``
    é uma relaxação válida quando calculada, ou None se recusado pelos limites.
    A soma e as podas usam Fractions exatos dos floats
    de entrada, evitando erro de arredondamento na certificação.
    """
    if tipos_por_local not in (1, 2):
        raise ValueError("tipos_por_local deve ser 1 ou 2.")
    if tempo_limite is None or not isfinite(tempo_limite) or tempo_limite <= 0:
        raise ValueError("tempo_limite deve ser finito e positivo.")
    if max_candidatos is not None and (not isinstance(max_candidatos, int) or max_candidatos < 0):
        raise ValueError("max_candidatos deve ser None ou inteiro não negativo.")
    if max_combinacoes is not None and (not isinstance(max_combinacoes, int) or max_combinacoes < 1):
        raise ValueError("max_combinacoes deve ser None ou inteiro positivo.")
    validar_instancia(instancia)
    n, m = len(instancia.candidatos), len(instancia.escolas)
    # Saturação evita construir inteiros enormes; None significa estimativa acima
    # do teto de contagem, não zero combinações.
    teto_estimativa = max_combinacoes or MAX_COMBINACOES_PADRAO
    estimativa: int | None = 1
    for _ in range(n):
        if estimativa > teto_estimativa // m:
            estimativa = None
            break
        estimativa *= m
    inicio = monotonic()
    if (max_candidatos is not None and n > max_candidatos) or (
        max_combinacoes is not None and (estimativa is None or estimativa > max_combinacoes)
    ):
        return ResultadoExato("limite", None, None, None, None, None, monotonic() - inicio, estimativa)

    linhas_exatas = tuple(tuple(Fraction.from_float(float(x)) for x in linha)
                          for linha in instancia.matriz_distancias)
    minimos = tuple(min(linha) for linha in linhas_exatas)
    lb_exato = sum(minimos, Fraction(0))
    lb = float(lb_exato)

    tipos = tuple(c.tipo_prova for c in instancia.candidatos)
    capacidades = tuple(e.capacidade for e in instancia.escolas)
    atribuicoes = [-1] * n
    cargas = [0] * m
    tipos_local: list[set[str]] = [set() for _ in range(m)]
    melhor: tuple[int, ...] | None = None
    melhor_custo: float | None = None
    melhor_exato: Fraction | None = None
    interrompido = False

    sufixos = [Fraction(0)] * (n + 1)
    for indice in range(n - 1, -1, -1):
        sufixos[indice] = sufixos[indice + 1] + minimos[indice]

    def dfs(i: int, custo: Fraction) -> None:
        nonlocal melhor, melhor_custo, melhor_exato, interrompido
        if interrompido:
            return
        if monotonic() - inicio >= tempo_limite:
            interrompido = True
            return
        if melhor_exato is not None and custo + sufixos[i] >= melhor_exato:
            return
        if i == n:
            aloc = tuple(atribuicoes)
            verificar(instancia, aloc, tipos_por_local)
            total = avaliar(instancia, aloc)
            if melhor_exato is None or custo < melhor_exato:
                melhor, melhor_custo, melhor_exato = aloc, total, custo
            return
        linha = linhas_exatas[i]
        for j, distancia in enumerate(linha):
            if cargas[j] >= capacidades[j]:
                continue
            novos = tipos_local[j] | {tipos[i]}
            if len(novos) > tipos_por_local:
                continue
            atribuicoes[i] = j
            cargas[j] += 1
            antigos = tipos_local[j]
            tipos_local[j] = novos
            dfs(i + 1, custo + distancia)
            tipos_local[j] = antigos
            cargas[j] -= 1
            if interrompido:
                return

    dfs(0, Fraction(0))
    elapsed = monotonic() - inicio
    status = "limite" if interrompido else ("otimo" if melhor is not None else "inviavel")
    certified_bound = melhor_custo if status == "otimo" else None
    gap = 0.0 if status == "otimo" else None
    return ResultadoExato(status, melhor, melhor_custo, certified_bound, lb, gap, elapsed, estimativa)
