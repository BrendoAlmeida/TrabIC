"""CLI de força bruta exata para instâncias pequenas."""

from __future__ import annotations

import argparse
import json

from src.instancias.io import ler_instancia
from src.otimizacao.exato_salas import resolver_exato_salas
from src.otimizacao.salas import resumo_salas


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=("Força bruta exata para instâncias pequenas; para milhares de candidatos "
                     "use instâncias menores ou um futuro MILP (ainda não implementado).")
    )
    parser.add_argument("--instancia", required=True)
    parser.add_argument("--tipos-por-local", required=True, type=int)
    parser.add_argument("--tempo-limite", type=float, default=10.0)
    args = parser.parse_args(argv)
    try:
        instancia = ler_instancia(args.instancia)
        resultado = resolver_exato_salas(instancia, args.tipos_por_local, tempo_limite=args.tempo_limite)
    except (ValueError, OSError) as erro:
        parser.error(str(erro))
    print(json.dumps({
        "status": resultado.status,
        "unidade": instancia.unidade_distancias,
        "tempo_segundos": resultado.tempo_segundos,
        "custo": resultado.custo,
        "alocacao": resultado.solucao.alocacao if resultado.solucao else None,
        "salas": [vars(sala) for sala in resumo_salas(instancia, resultado.solucao, args.tipos_por_local)] if resultado.solucao else None,
        "bound": resultado.bound,
        "lower_bound": resultado.lower_bound,
        "gap": resultado.gap,
    }, ensure_ascii=False))
    return 0 if resultado.status in ("otimo", "inviavel") else 2


if __name__ == "__main__":
    raise SystemExit(main())
