"""Interface integrada da heurística com alocação explícita por sala."""
from dataclasses import dataclass

from .busca_local import vnd
from .construtivo import construir
from .salas import SolucaoSalas, avaliar_salas, verificar_salas


@dataclass(frozen=True)
class ResultadoHeuristica:
    status: str
    alocacao: tuple[int, ...]
    solucao: SolucaoSalas | None
    custo_inicial: float | None
    custo_final: float | None
    motivo: str


def otimizar(instancia, tipos_por_local, seed=0, *, reinicios=10, alpha=0.2,
             limite_reparacao=100, limite_configuracoes=30000,
             max_avaliacoes=100000, tempo_limite=None):
    """Constrói e melhora uma solução; ``solucao`` contém os índices das salas.

    ``alocacao`` é uma projeção para escolas compatível com código antigo;
    ela isoladamente não certifica as restrições por sala. Falha heurística não
    constitui prova de inviabilidade. ``tempo_limite`` limita aproximadamente
    apenas a busca VND, não a construção nem o tempo total.
    """
    resultado = construir(instancia, tipos_por_local, seed, alpha=alpha,
                          reinicios=reinicios, limite_reparacao=limite_reparacao,
                          limite_configuracoes=limite_configuracoes)
    if resultado.solucao is None:
        return ResultadoHeuristica(resultado.status, (), None, None, None, resultado.motivo)
    busca = vnd(instancia, resultado.solucao, tipos_por_local, seed,
                max_avaliacoes=max_avaliacoes, tempo_limite=tempo_limite)
    verificar_salas(instancia, busca.solucao, tipos_por_local)
    return ResultadoHeuristica('viavel', tuple(j for j, _ in busca.solucao.alocacao),
                              busca.solucao, busca.custo_inicial,
                              avaliar_salas(instancia, busca.solucao), busca.motivo_parada)
