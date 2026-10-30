"""Conversões de matriz para computação numérica (não implementa solver).

O suporte JAX ativa ``jax_enable_x64`` para a precisão pedida. Essa configuração
é global ao processo JAX e afeta arrays criados posteriormente por JAX.
"""

from __future__ import annotations

from .models import Instancia


def matriz_distancias_numpy(instancia: Instancia):
    """Retorna a matriz da instância como array NumPy float64 em CPU."""
    try:
        import numpy as np
    except ImportError as erro:
        raise ImportError("NumPy é necessário; instale a dependência principal do projeto.") from erro
    return np.asarray(instancia.matriz_distancias, dtype=np.float64)


def matriz_distancias_jax(instancia: Instancia, dispositivo: str = "cpu"):
    """Converte a matriz em array JAX float64 no dispositivo ``cpu`` ou ``gpu``.

    JAX é importado apenas nesta chamada. GPU ausente/incompatível produz erro,
    sem fallback para CPU.
    """
    if dispositivo not in {"cpu", "gpu"}:
        raise ValueError("dispositivo deve ser 'cpu' ou 'gpu'.")
    try:
        import jax
        import jax.numpy as jnp
    except ImportError as erro:
        raise ImportError("JAX é opcional; instale as dependências de requirements-jax.txt.") from erro

    try:
        jax.config.update("jax_enable_x64", True)
    except Exception as erro:
        raise RuntimeError("Não foi possível habilitar precisão JAX float64 (jax_enable_x64).") from erro

    plataforma = "gpu" if dispositivo == "gpu" else "cpu"
    try:
        dispositivos = jax.devices(plataforma)
    except RuntimeError as erro:
        raise RuntimeError(f"Dispositivo JAX '{plataforma}' indisponível; nenhum fallback foi aplicado.") from erro
    if not dispositivos:
        raise RuntimeError(f"Dispositivo JAX '{plataforma}' indisponível; nenhum fallback foi aplicado.")
    return jax.device_put(jnp.asarray(instancia.matriz_distancias, dtype=jnp.float64), dispositivos[0])
