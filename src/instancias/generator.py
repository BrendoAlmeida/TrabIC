"""Geração reproduzível das duas instâncias da primeira entrega."""

import math
import random
from pathlib import Path

from .io import salvar_instancia, validar_instancia
from .models import Candidato, ErroInstancia, Escola, Instancia, TIPOS_PROVA


# Coordenadas sintéticas em um retângulo aproximado da região do Modelo.txt.
LIMITES_LATITUDE = (-21.805, -21.728)
LIMITES_LONGITUDE = (-43.404, -43.268)
CAPACIDADES_SALAS = ((16, 15),) * 12
DISTRIBUICAO_CANDIDATOS = {
    tipo: 53 if indice < 4 else 52
    for indice, tipo in enumerate(TIPOS_PROVA)
}
CENTROS_DOS_PARES = (
    (-21.782, -43.379),
    (-21.768, -43.340),
    (-21.750, -43.300),
)


def gerar_instancias(diretorio_saida: str | Path, semente: int = 2026) -> tuple[Path, Path]:
    """Gera e salva duas instâncias sem identificar a diferenciada pelos nomes."""
    pasta = Path(diretorio_saida)
    primeira = criar_instancia_base(semente)
    segunda = criar_instancia_diferenciada(semente + 1)
    validar_instancia_gerada(primeira)
    validar_instancia_gerada(segunda)

    caminho_primeira = pasta / "instancia_1.txt"
    caminho_segunda = pasta / "instancia_2.txt"
    salvar_instancia(primeira, caminho_primeira)
    salvar_instancia(segunda, caminho_segunda)
    return caminho_primeira, caminho_segunda


def criar_instancia_base(semente: int = 2026) -> Instancia:
    """Cria uma instância-base com distribuição geográfica espalhada."""
    return _montar_instancia(DISTRIBUICAO_CANDIDATOS, semente, agrupar_por_tipo=False)


def criar_instancia_diferenciada(semente: int = 2027) -> Instancia:
    """Cria grupos geográficos de tipos próximos com escolas entre eles."""
    return _montar_instancia(DISTRIBUICAO_CANDIDATOS, semente, agrupar_por_tipo=True)


def validar_instancia_gerada(instancia: Instancia) -> None:
    """Confere as regras da atividade e a possibilidade de alocar por tipo."""
    validar_instancia(instancia)
    quantidade_escolas = len(instancia.escolas)
    capacidade_total = sum(escola.capacidade for escola in instancia.escolas)
    quantidade_candidatos = len(instancia.candidatos)
    if not 10 <= quantidade_escolas <= 20:
        raise ErroInstancia("A quantidade de escolas deve ficar entre 10 e 20.")
    if not 200 <= capacidade_total <= 500:
        raise ErroInstancia("A capacidade total deve ficar entre 200 e 500 lugares.")
    utilizacao = quantidade_candidatos / capacidade_total
    if not 0.80 <= utilizacao <= 0.95:
        raise ErroInstancia("A quantidade de candidatos deve representar de 80% a 95% da capacidade.")

    if set(candidato.tipo_prova for candidato in instancia.candidatos) != set(TIPOS_PROVA):
        raise ErroInstancia("A instância deve incluir candidatos de todos os tipos de prova.")


def _montar_instancia(
    distribuicao_candidatos: dict[str, int],
    semente: int,
    agrupar_por_tipo: bool,
) -> Instancia:
    aleatorio = random.Random(semente)
    tipos_candidatos = [tipo for tipo, quantidade in distribuicao_candidatos.items() for _ in range(quantidade)]
    aleatorio.shuffle(tipos_candidatos)

    candidatos_lista = []
    for indice, tipo in enumerate(tipos_candidatos, start=1):
        if agrupar_por_tipo:
            latitude, longitude = _coordenada_candidato_agrupado(tipo, aleatorio)
        else:
            latitude = aleatorio.uniform(*LIMITES_LATITUDE)
            longitude = aleatorio.uniform(*LIMITES_LONGITUDE)
        candidatos_lista.append(
            Candidato(
                id=f"C{indice}",
                cep=f"{indice:08d}",
                tipo_prova=tipo,
                latitude=latitude,
                longitude=longitude,
            )
        )
    candidatos = tuple(candidatos_lista)

    if agrupar_por_tipo:
        posicoes_escolas = _posicoes_escolas_entre_grupos(aleatorio)
    else:
        posicoes_escolas = tuple(
            (aleatorio.uniform(*LIMITES_LATITUDE), aleatorio.uniform(*LIMITES_LONGITUDE))
            for _ in CAPACIDADES_SALAS
        )
    escolas = tuple(
        Escola(
            id=f"E{indice}",
            cep=f"{90000000 + indice:08d}",
            latitude=latitude,
            longitude=longitude,
            capacidades_salas=capacidades,
        )
        for indice, (capacidades, (latitude, longitude)) in enumerate(
            zip(CAPACIDADES_SALAS, posicoes_escolas), start=1
        )
    )
    matriz = tuple(
        tuple(_distancia_km(candidato, escola) for escola in escolas)
        for candidato in candidatos
    )
    return Instancia(candidatos, escolas, matriz)


def _coordenada_candidato_agrupado(tipo: str, aleatorio: random.Random) -> tuple[float, float]:
    indice_tipo = TIPOS_PROVA.index(tipo)
    indice_par = indice_tipo // 2
    lado = -1 if indice_tipo % 2 == 0 else 1
    latitude_centro, longitude_centro = CENTROS_DOS_PARES[indice_par]
    latitude = aleatorio.gauss(latitude_centro, 0.0025)
    longitude = aleatorio.gauss(longitude_centro + lado * 0.004, 0.0035)
    return (
        min(max(latitude, LIMITES_LATITUDE[0]), LIMITES_LATITUDE[1]),
        min(max(longitude, LIMITES_LONGITUDE[0]), LIMITES_LONGITUDE[1]),
    )


def _posicoes_escolas_entre_grupos(
    aleatorio: random.Random,
) -> tuple[tuple[float, float], ...]:
    """Põe duas escolas entre cada par de grupos e uma perto de cada grupo."""
    posicoes = []
    for latitude, longitude in CENTROS_DOS_PARES:
        posicoes.extend(
            [
                (latitude, longitude - 0.004),
                (latitude - 0.002, longitude),
                (latitude + 0.002, longitude),
                (latitude, longitude + 0.004),
            ]
        )
    return tuple(
        (
            latitude + aleatorio.gauss(0, 0.0005),
            longitude + aleatorio.gauss(0, 0.0005),
        )
        for latitude, longitude in posicoes
    )


def _distancia_km(candidato: Candidato, escola: Escola) -> float:
    """Calcula a distância em linha reta usando a fórmula de Haversine."""
    raio_terra_km = 6371.0
    lat1 = math.radians(candidato.latitude)
    lat2 = math.radians(escola.latitude)
    delta_lat = math.radians(escola.latitude - candidato.latitude)
    delta_lon = math.radians(escola.longitude - candidato.longitude)
    a = (
        math.sin(delta_lat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(delta_lon / 2) ** 2
    )
    return 2 * raio_terra_km * math.asin(min(1.0, math.sqrt(a)))
