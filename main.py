"""CLI CPU da heurística regret+VND."""

import argparse
import json
import subprocess
import sys
import time

from src.instancias import ler_instancia
from src.otimizacao.heuristica import otimizar
from src.otimizacao.modelo import avaliar, verificar


def _versao_git():
    try:
        rev = subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], check=True, capture_output=True, text=True).stdout)
        return {"commit": rev, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Executa heurística regret+VND em CPU.")
    parser.add_argument("--instancia", required=True)
    parser.add_argument("--tipos-por-local", required=True, type=int, choices=(1, 2))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--execucoes", type=int, default=1)
    args = parser.parse_args(argv)
    if args.execucoes < 1:
        parser.error("--execucoes deve ser positivo")
    try:
        instancia = ler_instancia(args.instancia)
        execucoes = []
        for idx in range(args.execucoes):
            seed = args.seed + idx
            inicio = time.perf_counter()
            r = otimizar(instancia, args.tipos_por_local, seed)
            decorrido = time.perf_counter() - inicio
            item = {"seed": seed, "status": r.status,
                    "custo_inicial": r.custo_inicial if r.status == "viavel" else None,
                    "custo_final": r.custo_final if r.status == "viavel" else None,
                    "tempo_segundos": decorrido,
                    "metodo_tempo": "time.perf_counter (tempo total CPU)",
                    "alocacao": list(r.alocacao)}
            if r.status == "viavel":
                verificar(instancia, r.alocacao, args.tipos_por_local)
                item["custo_reavaliado"] = avaliar(instancia, r.alocacao)
                item["distribuicao"] = {str(j): r.alocacao.count(j) for j in sorted(set(r.alocacao))}
            execucoes.append(item)
        print(json.dumps({"instancia": args.instancia, "unidade": instancia.unidade_distancias,
                          "tipos_por_local": args.tipos_por_local, "backend": "numpy/cpu",
                          "parametros": {"reinicios": 4, "max_iter": 20, "max_tentativas": 20000},
                          "versao": _versao_git(), "execucoes": execucoes}, ensure_ascii=False, allow_nan=False))
        return 0 if all(x["status"] == "viavel" for x in execucoes) else 2
    except (OSError, ValueError, RuntimeError) as erro:
        print(f"Erro: {erro}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
