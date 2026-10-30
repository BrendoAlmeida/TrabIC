"""Executa experiências reproduzíveis da heurística por sala em CPU.

O relógio inclui construção, busca, verificação e avaliação, mas não leitura e
serialização. O limite global é cooperativo: verificações ocorrem entre operações e podem ultrapassar o prazo.
"""
import argparse
from dataclasses import asdict
import hashlib
from fractions import Fraction
import math
import json
import math
import platform
from pathlib import Path
import statistics
import subprocess
import sys
import time

import numpy as np

from src.instancias import ler_instancia
from src.otimizacao.construtivo import construir
from src.otimizacao.construtivo_add import construir_add
from src.otimizacao.busca_local import vnd
from src.otimizacao.cadeias import buscar_cadeias
from src.otimizacao.salas import avaliar_salas, resumo_salas, verificar_salas


def _versao():
    raiz = Path(__file__).resolve().parent
    arquivos = [Path(__file__), *(raiz / 'src/otimizacao' / nome for nome in
                                ('salas.py', 'construtivo.py', 'construtivo_add.py', 'busca_local.py', 'cadeias.py')),
                *(raiz / 'src/instancias' / nome for nome in ('__init__.py', 'models.py', 'io.py'))]
    fontes = {str(p.relative_to(raiz)): hashlib.sha256(p.read_bytes()).hexdigest() for p in arquivos}
    try:
        commit = subprocess.run(['git', '-C', str(raiz), 'rev-parse', 'HEAD'],
                                capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    return {'git_head': commit, 'sha256_fontes': fontes}


def limite_inferior_distancia(instancia):
    """Soma exata dos mínimos float, arredondada para baixo ao representar.

    A análise é executada fora dos tempos de cada tentativa.
    """
    if any(not linha for linha in instancia.matriz_distancias):
        return None
    exato = sum((Fraction.from_float(float(min(linha)))
                 for linha in instancia.matriz_distancias), Fraction())
    resultado = float(exato)
    if Fraction.from_float(resultado) > exato:
        resultado = math.nextafter(resultado, -math.inf)
    return resultado


def executar(instancia, *, tipos_por_local, seeds, metodo="regret", alpha=0.2, reinicios=10,
             limite_reparacao=100, limite_configuracoes=30000,
             max_avaliacoes=100000, tempo_busca=None, ablacao="vnd",
             cadeias=False, comprimento_cadeia=3, avaliacoes_cadeias=10000,
             tempo_cadeias=None, tempo_total=None):
    """Retorna registros e estatísticas; custo só é registrado após verificar salas."""
    if tempo_total is not None and (not math.isfinite(tempo_total) or tempo_total < 0):
        raise ValueError('tempo_total deve ser finito e não negativo')
    seeds = list(seeds)
    if not seeds:
        raise ValueError('É necessário fornecer pelo menos uma seed.')
    metodos = {'regret': construir, 'add': construir_add}
    if metodo not in metodos:
        raise ValueError('Método desconhecido.')
    opcoes_ablacao = {'construtivo': (), 'shift': ('shift',),
                       'shift_swap': ('shift', 'swap'),
                       'vnd': ('shift', 'swap', 'reconfiguracao')}
    if ablacao not in opcoes_ablacao:
        raise ValueError('Ablação desconhecida.')
    limite_inferior = limite_inferior_distancia(instancia)
    registros = []
    for seed in seeds:
        inicio = time.perf_counter()
        prazo = None if tempo_total is None else inicio + tempo_total
        inicial = metodos[metodo](instancia, tipos_por_local, seed, alpha=alpha,
                            reinicios=reinicios, limite_reparacao=limite_reparacao,
                            limite_configuracoes=limite_configuracoes, prazo=prazo)
        fim_construcao = time.perf_counter()
        registro = {'seed': seed, 'status': inicial.status, 'motivo': inicial.motivo,
                    'tentativas': inicial.tentativas, 'reparacoes': inicial.reparacoes,
                    'tempo_construcao_s': fim_construcao - inicio,
                    'limite_global_atingido': False,
                    'tempo_busca_s': 0.0, 'tempo_cadeias_s': 0.0, 'cadeias': None,
                    'custo_inicial': None, 'custo_final': None,
                    'melhoria': None, 'gap_limite_inferior_pct': None, 'busca': None, 'escolas_utilizadas': None,
                    'salas_utilizadas': None, 'salas': None}
        if inicial.solucao is not None:
            verificar_salas(instancia, inicial.solucao, tipos_por_local)
            custo_inicial = avaliar_salas(instancia, inicial.solucao)
            if ablacao == 'construtivo':
                solucao_final = inicial.solucao
                dados_busca = None
            else:
                inicio_busca = time.perf_counter()
                restante = None if prazo is None else max(0.0, prazo - time.perf_counter())
                limite_etapa = tempo_busca if restante is None else (restante if tempo_busca is None else min(restante, tempo_busca))
                busca = vnd(instancia, inicial.solucao, tipos_por_local, seed,
                            max_avaliacoes=max_avaliacoes, tempo_limite=limite_etapa,
                            vizinhancas=opcoes_ablacao[ablacao])
                registro['tempo_busca_s'] = time.perf_counter() - inicio_busca
                solucao_final = busca.solucao
                dados_busca = {'avaliacoes': busca.avaliacoes, 'melhorias': busca.melhorias,
                               'motivo_parada': busca.motivo_parada}
            if cadeias:
                inicio_cadeias = time.perf_counter()
                restante = None if prazo is None else max(0.0, prazo - time.perf_counter())
                limite_etapa = tempo_cadeias if restante is None else (restante if tempo_cadeias is None else min(restante, tempo_cadeias))
                resultado_cadeias = buscar_cadeias(
                    instancia, solucao_final, tipos_por_local, seed,
                    max_comprimento=comprimento_cadeia, max_avaliacoes=avaliacoes_cadeias,
                    tempo_limite=limite_etapa)
                registro['tempo_cadeias_s'] = time.perf_counter() - inicio_cadeias
                solucao_final = resultado_cadeias.solucao
                registro['cadeias'] = {'avaliacoes': resultado_cadeias.avaliacoes,
                                       'melhorias': resultado_cadeias.melhorias,
                                       'custo_inicial': resultado_cadeias.custo_inicial,
                                       'custo_final': resultado_cadeias.custo_final,
                                       'motivo_parada': resultado_cadeias.motivo_parada}
            verificar_salas(instancia, solucao_final, tipos_por_local)
            custo_final = avaliar_salas(instancia, solucao_final)
            salas = resumo_salas(instancia, solucao_final, tipos_por_local)
            registro.update(custo_inicial=custo_inicial, custo_final=custo_final,
                            melhoria=custo_inicial - custo_final,
                            gap_limite_inferior_pct=(100 * (custo_final - limite_inferior) / limite_inferior
                                                    if limite_inferior is not None and limite_inferior > 0 else None),
                            busca=dados_busca,
                            escolas_utilizadas=len({s.indice_escola for s in salas if s.tipo is not None}),
                            salas_utilizadas=sum(s.tipo is not None for s in salas),
                            salas=[asdict(s) for s in salas])
        registro['tempo_total_s'] = time.perf_counter() - inicio
        registro['limite_global_atingido'] = prazo is not None and time.perf_counter() >= prazo
        registros.append(registro)
    custos = [r['custo_final'] for r in registros if r['custo_final'] is not None]
    resumo = {'execucoes': len(registros), 'sucessos': len(custos),
              'falhas': len(registros) - len(custos), 'taxa_sucesso': len(custos) / len(registros),
              'melhor': min(custos) if custos else None,
              'media': statistics.mean(custos) if custos else None,
              'desvio_padrao_populacional': statistics.pstdev(custos) if custos else None,
              'pior': max(custos) if custos else None,
              'limite_inferior': limite_inferior,
              'gap_definicao': '100*(custo-limite_inferior)/limite_inferior; null se limite <= 0 ou inexistente',
              'tempo_medio_s': statistics.mean(r['tempo_total_s'] for r in registros),
              'tempo_medio_construcao_s': statistics.mean(r['tempo_construcao_s'] for r in registros),
              'tempo_medio_busca_s': statistics.mean(r['tempo_busca_s'] for r in registros),
              'tempo_medio_cadeias_s': statistics.mean(r['tempo_cadeias_s'] for r in registros)}
    return {'parametros': {'metodo': metodo, 'tipos_por_local': tipos_por_local, 'seeds': seeds, 'alpha': alpha,
                           'reinicios': reinicios, 'limite_reparacao': limite_reparacao,
                           'limite_configuracoes': limite_configuracoes,
                           'max_avaliacoes': max_avaliacoes, 'tempo_busca_s': tempo_busca, 'tempo_total_s': tempo_total, 'ablacao': ablacao,
                           'cadeias': cadeias, 'comprimento_cadeia': comprimento_cadeia,
                           'avaliacoes_cadeias': avaliacoes_cadeias, 'tempo_cadeias_s': tempo_cadeias},
            'metadados': {'versao': _versao(), 'python': platform.python_version(),
                          'numpy': np.__version__, 'plataforma': platform.platform(),
                          'processador': platform.processor(), 'backend': 'numpy',
                          'dispositivo': 'cpu', 'unidade_distancias': instancia.unidade_distancias,
                          'cronometro': 'time.perf_counter; inclui construção, busca, verificação e avaliação; exclui leitura, cálculo do limite inferior e serialização'},
            'resumo': resumo, 'execucoes': registros}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--instancia', required=True)
    parser.add_argument('--tipos-por-local', type=int, required=True)
    parser.add_argument('--seed', type=int, default=42, help='Primeira seed; sementes consecutivas.')
    parser.add_argument('--execucoes', type=int, default=10)
    parser.add_argument('--metodo', choices=('regret', 'add'), default='regret')
    parser.add_argument('--ablacao', choices=('construtivo', 'shift', 'shift_swap', 'vnd'), default='vnd')
    parser.add_argument('--alpha', type=float, default=0.2)
    parser.add_argument('--reinicios', type=int, default=10)
    parser.add_argument('--limite-reparacao', type=int, default=100)
    parser.add_argument('--limite-configuracoes', type=int, default=30000)
    parser.add_argument('--max-avaliacoes', type=int, default=100000)
    parser.add_argument('--tempo-total', type=float, default=None,
                        help='Limite cooperativo em segundos por execução, incluindo construção e busca.')
    parser.add_argument('--tempo-busca', type=float, default=None,
                        help='Limite aproximado em segundos apenas da busca local.')
    parser.add_argument('--cadeias', action='store_true', help='Executar cadeias após a busca selecionada.')
    parser.add_argument('--comprimento-cadeia', type=int, default=3)
    parser.add_argument('--avaliacoes-cadeias', type=int, default=10000)
    parser.add_argument('--tempo-cadeias', type=float, default=None, help='Limite aproximado em segundos apenas para cadeias.')
    parser.add_argument('--saida', required=True, help='Arquivo JSON novo, sem sobrescrever.')
    args = parser.parse_args(argv)
    if args.execucoes < 1:
        parser.error('--execucoes deve ser positivo')
    if Path(args.saida).exists():
        parser.error('--saida já existe; escolha outro caminho')
    caminho = Path(args.instancia)
    instancia = ler_instancia(caminho)
    resultado = executar(instancia, tipos_por_local=args.tipos_por_local,
                         seeds=range(args.seed, args.seed + args.execucoes), metodo=args.metodo, alpha=args.alpha,
                         reinicios=args.reinicios, limite_reparacao=args.limite_reparacao,
                         limite_configuracoes=args.limite_configuracoes,
                         max_avaliacoes=args.max_avaliacoes, tempo_busca=args.tempo_busca, ablacao=args.ablacao,
                         cadeias=args.cadeias, comprimento_cadeia=args.comprimento_cadeia,
                         avaliacoes_cadeias=args.avaliacoes_cadeias, tempo_cadeias=args.tempo_cadeias,
                         tempo_total=args.tempo_total)
    resultado['metadados']['instancia'] = str(caminho)
    resultado['metadados']['sha256_instancia'] = hashlib.sha256(caminho.read_bytes()).hexdigest()
    with open(args.saida, 'x', encoding='utf-8') as arquivo:
        json.dump(resultado, arquivo, ensure_ascii=False, indent=2, allow_nan=False)
        arquivo.write('\n')
    print(json.dumps(resultado['resumo'], ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    sys.exit(main())
