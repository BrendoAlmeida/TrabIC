"""Gráficos interativos para explorar uma instância no notebook."""

from collections import Counter

import plotly.graph_objects as go

from .models import Instancia, TIPOS_PROVA


CORES_TIPOS = {
    "M1": "#636EFA",
    "M2": "#EF553B",
    "M3e": "#00CC96",
    "M3s": "#AB63FA",
    "M3h": "#FFA15A",
    "M3d": "#19D3F3",
}


def grafico_instancia(instancia: Instancia, titulo: str = "Instância") -> go.Figure:
    """Mostra candidatos coloridos por tipo e escolas identificadas no mapa."""
    figura = go.Figure()

    for tipo in TIPOS_PROVA:
        indices = [
            indice
            for indice, candidato in enumerate(instancia.candidatos)
            if candidato.tipo_prova == tipo
        ]
        if not indices:
            continue

        candidatos = [instancia.candidatos[indice] for indice in indices]
        detalhes = []
        for indice, candidato in zip(indices, candidatos):
            distancias = instancia.matriz_distancias[indice]
            indice_escola_proxima = min(range(len(distancias)), key=distancias.__getitem__)
            escola_proxima = instancia.escolas[indice_escola_proxima]
            detalhes.append(
                [
                    candidato.id,
                    candidato.cep,
                    candidato.tipo_prova,
                    candidato.latitude,
                    candidato.longitude,
                    escola_proxima.id,
                    distancias[indice_escola_proxima],
                ]
            )

        figura.add_trace(
            go.Scatter(
                x=[candidato.longitude for candidato in candidatos],
                y=[candidato.latitude for candidato in candidatos],
                mode="markers",
                name=tipo,
                marker={"size": 8, "color": CORES_TIPOS[tipo], "opacity": 0.72},
                customdata=detalhes,
                hovertemplate=(
                    "Candidato: %{customdata[0]}<br>"
                    "Tipo de prova: %{customdata[2]}<br>"
                    "CEP: %{customdata[1]}<br>"
                    "Latitude: %{customdata[3]:.6f}<br>"
                    "Longitude: %{customdata[4]:.6f}<br>"
                    "Escola mais próxima por distância: %{customdata[5]}<br>"
                    "Distância: %{customdata[6]:.2f} km"
                    "<extra></extra>"
                ),
            )
        )

    detalhes_escolas = [
        [
            escola.id,
            escola.cep,
            len(escola.capacidades_salas),
            "; ".join(str(capacidade) for capacidade in escola.capacidades_salas),
            escola.capacidade,
            escola.latitude,
            escola.longitude,
        ]
        for escola in instancia.escolas
    ]
    figura.add_trace(
        go.Scatter(
            x=[escola.longitude for escola in instancia.escolas],
            y=[escola.latitude for escola in instancia.escolas],
            mode="markers+text",
            name="Escolas",
            text=[escola.id for escola in instancia.escolas],
            textposition="top center",
            marker={"size": 15, "color": "#222222", "symbol": "diamond", "line": {"color": "white", "width": 1}},
            customdata=detalhes_escolas,
            hovertemplate=(
                "Escola: %{customdata[0]}<br>"
                "CEP: %{customdata[1]}<br>"
                "Salas: %{customdata[2]}<br>"
                "Capacidades das salas: %{customdata[3]}<br>"
                "Capacidade total: %{customdata[4]}<br>"
                "Latitude: %{customdata[5]:.6f}<br>"
                "Longitude: %{customdata[6]:.6f}"
                "<extra></extra>"
            ),
        )
    )

    figura.update_layout(
        title={
            "text": (
                f"{titulo}<br><sup>Candidatos por tipo de prova; escolas sem tipo predefinido</sup>"
            )
        },
        xaxis_title="Longitude",
        yaxis_title="Latitude",
        hovermode="closest",
        legend_title="Tipo de prova",
        height=720,
        template="plotly_white",
        margin={"l": 60, "r": 30, "t": 90, "b": 55},
    )
    return figura


def grafico_candidatos_por_tipo(instancia: Instancia, titulo: str = "Candidatos por tipo") -> go.Figure:
    """Compara a quantidade de candidatos de cada tipo de prova."""
    contagens = Counter(candidato.tipo_prova for candidato in instancia.candidatos)
    figura = go.Figure(
        go.Bar(
            x=list(TIPOS_PROVA),
            y=[contagens[tipo] for tipo in TIPOS_PROVA],
            marker_color=[CORES_TIPOS[tipo] for tipo in TIPOS_PROVA],
            text=[contagens[tipo] for tipo in TIPOS_PROVA],
            textposition="outside",
            hovertemplate="Tipo: %{x}<br>Candidatos: %{y}<extra></extra>",
        )
    )
    figura.update_layout(
        title=titulo,
        xaxis_title="Tipo de prova",
        yaxis_title="Quantidade de candidatos",
        template="plotly_white",
        showlegend=False,
        height=400,
    )
    return figura
