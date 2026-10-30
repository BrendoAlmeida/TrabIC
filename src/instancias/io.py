"""Leitura e escrita do formato textual usado nos exemplos."""

import math
import re
from pathlib import Path

from .models import Candidato, ErroInstancia, Escola, Instancia, TIPOS_PROVA


def carregar_instancia(caminho: str | Path) -> Instancia:
    """Carrega uma instância no formato do Modelo.txt."""
    linhas = Path(caminho).read_text(encoding="utf-8").splitlines()
    secoes: dict[str, list[str]] = {"candidatos": [], "escolas": [], "distancias": []}
    secao_atual: str | None = None
    quantidade_candidatos: int | None = None
    quantidade_escolas: int | None = None
    quantidades_declaradas: set[str] = set()
    unidade_distancias = "desconhecida"

    conteudo = [linha.strip() for linha in linhas if linha.strip()]
    if len(conteudo) == 1 and _parece_caminho(conteudo[0]):
        raise ErroInstancia("O arquivo contém apenas um caminho para outra instância; caminhos externos não são seguidos.")

    for numero_linha, linha_original in enumerate(linhas, start=1):
        linha = linha_original.strip()
        if not linha:
            continue
        if linha.startswith("NUM_CANDIDATOS:") or linha.startswith("NUM_ESCOLAS:"):
            # G6 declara as duas quantidades na mesma linha; também aceitamos
            # os cabeçalhos tradicionais, um por linha.
            encontrados = []
            posicao = 0
            while posicao < len(linha):
                if linha[posicao:].strip() == "":
                    posicao = len(linha)
                    break
                declaracao = re.match(r"\s*(NUM_CANDIDATOS|NUM_ESCOLAS):\s*([+-]?\d+)(?=\s|$)", linha[posicao:])
                if declaracao is None:
                    raise ErroInstancia(f"Declaração de quantidade inválida na linha {numero_linha}.")
                chave, valor = declaracao.groups()
                if chave in quantidades_declaradas or any(chave == chave_existente for chave_existente, _ in encontrados):
                    raise ErroInstancia(f"Declaração duplicada de {chave} na linha {numero_linha}.")
                encontrados.append((chave, valor))
                posicao += declaracao.end()
            for chave, valor in encontrados:
                quantidades_declaradas.add(chave)
                if chave == "NUM_CANDIDATOS":
                    quantidade_candidatos = _ler_inteiro(valor, numero_linha)
                else:
                    quantidade_escolas = _ler_inteiro(valor, numero_linha)
            continue
        if linha.startswith("# CANDIDATOS"):
            secao_atual = "candidatos"
            continue
        if linha.startswith("# ESCOLAS"):
            secao_atual = "escolas"
            continue
        if linha.startswith("# MATRIZ_DISTANCIAS"):
            secao_atual = "distancias"
            cabecalho = linha.lower()
            if "em km" in cabecalho:
                unidade_distancias = "km"
            elif "em graus" in cabecalho or "em grau" in cabecalho:
                unidade_distancias = "graus"
            continue
        if linha.startswith("#"):
            continue
        if secao_atual is None:
            raise ErroInstancia(f"Conteúdo fora de uma seção na linha {numero_linha}.")
        secoes[secao_atual].append(linha)

    if quantidade_candidatos is None or quantidade_escolas is None:
        raise ErroInstancia("O arquivo precisa declarar NUM_CANDIDATOS e NUM_ESCOLAS.")

    candidatos = tuple(_ler_candidato(linha, i) for i, linha in enumerate(secoes["candidatos"], 1))
    escolas = tuple(_ler_escola(linha, i) for i, linha in enumerate(secoes["escolas"], 1))
    matriz = tuple(_ler_linha_distancia(linha, i) for i, linha in enumerate(secoes["distancias"], 1))

    if len(candidatos) != quantidade_candidatos:
        raise ErroInstancia(
            f"NUM_CANDIDATOS declara {quantidade_candidatos}, mas há {len(candidatos)} registros."
        )
    if len(escolas) != quantidade_escolas:
        raise ErroInstancia(f"NUM_ESCOLAS declara {quantidade_escolas}, mas há {len(escolas)} registros.")
    instancia = Instancia(candidatos, escolas, matriz, unidade_distancias)
    validar_instancia(instancia)
    return instancia


def salvar_instancia(instancia: Instancia, caminho: str | Path) -> None:
    """Salva uma instância no formato textual do Modelo.txt."""
    validar_instancia(instancia)
    destino = Path(caminho)
    destino.parent.mkdir(parents=True, exist_ok=True)

    linhas = [
        f"NUM_CANDIDATOS: {len(instancia.candidatos)}",
        f"NUM_ESCOLAS: {len(instancia.escolas)}",
        "",
        "# CANDIDATOS (id, CEP, tipo_prova, latitude, longitude)",
    ]
    for candidato in instancia.candidatos:
        linhas.append(
            f"{candidato.id}, {candidato.cep}, {candidato.tipo_prova}, "
            f"{candidato.latitude:.6f}, {candidato.longitude:.6f}"
        )

    linhas.extend(["", "# ESCOLAS (id, CEP, latitude, longitude, num_salas, capacidades_salas)"])
    for escola in instancia.escolas:
        capacidades = ";".join(str(capacidade) for capacidade in escola.capacidades_salas)
        linhas.append(
            f"{escola.id}, {escola.cep}, {escola.latitude:.6f}, {escola.longitude:.6f}, "
            f"{len(escola.capacidades_salas)}, {capacidades}"
        )

    unidade = instancia.unidade_distancias
    anotacao = {"km": "km", "graus": "graus", "desconhecida": "desconhecida"}.get(unidade)
    if anotacao is None:
        raise ErroInstancia(f"Unidade de distâncias inválida: {unidade}.")
    linhas.extend(["", f"# MATRIZ_DISTANCIAS (linhas=candidatos, colunas=escolas, em {anotacao})"])
    linhas.extend(", ".join(f"{distancia:.4f}" for distancia in linha) for linha in instancia.matriz_distancias)
    destino.write_text("\n".join(linhas) + "\n", encoding="utf-8")


def validar_instancia(instancia: Instancia) -> None:
    """Confere consistência estrutural sem restringir os limites de uma atividade."""
    if not instancia.candidatos:
        raise ErroInstancia("A instância precisa ter ao menos um candidato.")
    if not instancia.escolas:
        raise ErroInstancia("A instância precisa ter ao menos uma escola.")
    _validar_ids((c.id for c in instancia.candidatos), "candidato")
    _validar_ids((e.id for e in instancia.escolas), "escola")

    for candidato in instancia.candidatos:
        if candidato.tipo_prova not in TIPOS_PROVA:
            raise ErroInstancia(f"Tipo de prova inválido para {candidato.id}: {candidato.tipo_prova}.")
        _validar_coordenadas(candidato.latitude, candidato.longitude, candidato.id)
    for escola in instancia.escolas:
        if not escola.capacidades_salas or any(capacidade <= 0 for capacidade in escola.capacidades_salas):
            raise ErroInstancia(f"Capacidades de sala inválidas para {escola.id}.")
        _validar_coordenadas(escola.latitude, escola.longitude, escola.id)

    if len(instancia.matriz_distancias) != len(instancia.candidatos):
        raise ErroInstancia("A matriz precisa ter uma linha por candidato.")
    for indice, linha in enumerate(instancia.matriz_distancias, start=1):
        if len(linha) != len(instancia.escolas):
            raise ErroInstancia(f"A linha {indice} da matriz precisa ter uma coluna por escola.")
        if any(not math.isfinite(distancia) or distancia < 0 for distancia in linha):
            raise ErroInstancia(f"A linha {indice} da matriz contém distância inválida.")


def _ler_candidato(linha: str, numero: int) -> Candidato:
    campos = [campo.strip() for campo in linha.split(",")]
    if len(campos) != 5:
        raise ErroInstancia(f"Registro de candidato inválido na seção, linha {numero}.")
    try:
        return Candidato(campos[0], campos[1], campos[2], float(campos[3]), float(campos[4]))
    except ValueError as erro:
        raise ErroInstancia(f"Valor inválido no candidato da linha {numero}: {erro}.") from erro


def _ler_escola(linha: str, numero: int) -> Escola:
    campos = [campo.strip() for campo in linha.split(",")]
    if len(campos) != 6:
        raise ErroInstancia(f"Registro de escola inválido na seção, linha {numero}.")
    try:
        latitude = float(campos[2])
        longitude = float(campos[3])
        numero_salas = int(campos[4])
        capacidades = tuple(int(valor.strip()) for valor in campos[5].split(";"))
    except ValueError as erro:
        raise ErroInstancia(f"Valor inválido na escola da linha {numero}: {erro}.") from erro
    if len(capacidades) != numero_salas:
        raise ErroInstancia(
            f"A escola da linha {numero} declara {numero_salas} salas, "
            f"mas lista {len(capacidades)} capacidades."
        )
    return Escola(campos[0], campos[1], latitude, longitude, capacidades)


def _ler_linha_distancia(linha: str, numero: int) -> tuple[float, ...]:
    try:
        return tuple(float(valor.strip()) for valor in linha.split(","))
    except ValueError as erro:
        raise ErroInstancia(f"Distância inválida na linha {numero}: {erro}.") from erro


def _ler_inteiro(valor: str, numero_linha: int) -> int:
    try:
        quantidade = int(valor.strip())
    except ValueError as erro:
        raise ErroInstancia(f"Quantidade inválida na linha {numero_linha}.") from erro
    if quantidade < 0:
        raise ErroInstancia(f"Quantidade negativa na linha {numero_linha}.")
    return quantidade


def _parece_caminho(texto: str) -> bool:
    """Detecta apenas sintaxes explícitas de caminho, não texto arbitrário."""
    return (
        texto.startswith(("/", "./", "../", "~/", "\\\\", ".\\", "..\\"))
        or bool(re.match(r"^[A-Za-z]:[\\/]", texto))
    )


# Interface estável em português; manter carregar_instancia por compatibilidade.
ler_instancia = carregar_instancia


def _validar_ids(ids, elemento: str) -> None:
    lista = list(ids)
    if any(not identificador for identificador in lista) or len(lista) != len(set(lista)):
        raise ErroInstancia(f"Os identificadores de {elemento} devem ser preenchidos e únicos.")


def _validar_coordenadas(latitude: float, longitude: float, identificador: str) -> None:
    if not -90 <= latitude <= 90 or not -180 <= longitude <= 180:
        raise ErroInstancia(f"Coordenadas inválidas para {identificador}.")
