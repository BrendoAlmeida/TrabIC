"""Comparação de tabu e multi-start com parada por estagnação e teto rígido."""
import os
for name in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS',
             'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[name] = '1'

import argparse
import csv
import hashlib
import json
import platform
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.instancias import ler_instancia
from src.otimizacao.busca_lns import busca_lns
from src.otimizacao.busca_perfil_fluxo import busca_perfil_fluxo
from src.otimizacao.busca_multiexchange import busca_multiexchange
from src.otimizacao.busca_tabu import busca_tabu
from src.otimizacao.busca_ciclos_salas import busca_ciclos_salas
from src.otimizacao.busca_troca_salas import busca_troca_salas
from src.otimizacao.construtivo import construir
from src.otimizacao.construtivo_add import construir_add
from src.otimizacao.fluxo_perfil import otimizar_perfil
from src.otimizacao.salas import avaliar_salas, verificar_salas

INSTANCES = ('Instancia1_G10.txt', 'Instancia1_G5.txt', 'Instancia2_G6.txt')
METHODS = ('regret_tabu', 'multistart_tabu', 'multistart_tabu_multiexchange',
           'multistart_tabu_room_exchange', 'multistart_add_tabu',
           'multistart_tabu_room_cycle', 'multistart_tabu_lns',
           'multistart_tabu_flow', 'multistart_tabu_lns_flow',
           'multistart_tabu_profile_flow', 'multistart_tabu_profile_flow_lb',
           'multistart_tabu_profile_flow_costguided')
SEEDS = tuple(range(42, 47))
STAGNATION_SECONDS = 30.0
MAX_SECONDS = 650.0
CHUNK_SECONDS = 5.0
ROOM_CYCLE_CHUNK_SECONDS = 10.0
FLOW_CYCLE_SECONDS = 45.0
FLOW_CONSTRUCTOR_SECONDS = 10.0
LNS_FLOW_CYCLE_SECONDS = 45.0
PROFILE_FLOW_CYCLE_SECONDS = 45.0
PROFILE_FLOW_GUIDED_CYCLE_SECONDS = 75.0
PROFILE_FLOW_COST_GUIDED_CYCLE_SECONDS = 60.0
MAX_EVALS = None
TOLERANCE = 1e-10


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _save(path, payload):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2,
                               allow_nan=False) + '\n', encoding='utf-8')
    os.replace(temp, path)


def _search(inst, initial, k, seed, seconds, stagnation):
    if seconds <= 0:
        return None
    return busca_tabu(inst, initial, k, seed, tempo_limite=seconds,
                      tempo_sem_melhoria=max(1e-6, stagnation),
                      max_avaliacoes=MAX_EVALS)


def execute(job):
    """Executa um método até 30 s sem novo melhor ou até o teto rígido."""
    path = ROOT / 'data/instancias' / job['instance']
    if sha256(path) != job['instance_sha256']:
        raise RuntimeError('A instância mudou durante o experimento.')
    inst = ler_instancia(path)
    inicio = time.perf_counter()
    elapsed_before = float(job.get('resume_prior_elapsed_s', 0.0))
    remaining_budget = float(job.get('remaining_budget_s', MAX_SECONDS))
    deadline = inicio + remaining_budget
    melhor = None
    melhor_custo = None
    instante_melhor = None
    ultima_melhoria = None
    motivo = f'teto_{int(MAX_SECONDS)}s'
    construcoes = avaliacoes_tabu = melhorias_tabu = ciclos = 0
    ciclos_exchange = movimentos_exchange = avaliacoes_exchange = 0
    trocas_salas = pares_salas = 0
    ciclos_quartos = quartos_movidos = candidatos_movidos = avaliacoes_ciclos = 0
    iteracoes_lns = reparos_lns = reparos_lns_falhos = melhorias_lns = 0
    tempo_fluxo = 0.0
    tipos_otimizados_fluxo = aumentos_fluxo = relaxacoes_fluxo = 0
    fluxos_lns = tipos_fluxo_lns = aumentos_fluxo_lns = relaxacoes_fluxo_lns = 0
    perfis_propostos = perfis_viaveis = perfis_rejeitados = 0
    fluxos_perfil = tipos_fluxo_perfil = aumentos_fluxo_perfil = relaxacoes_fluxo_perfil = 0
    validacoes = 0
    checkpoint = Path(job['output']).with_suffix('.checkpoint')

    def elapsed_total():
        return elapsed_before + time.perf_counter() - inicio

    if elapsed_before > 0:
        if not checkpoint.exists():
            raise RuntimeError('Checkpoint de retomada ausente.')
        saved = json.loads(checkpoint.read_text(encoding='utf-8'))
        for key in ('instance', 'instance_sha256', 'k', 'method', 'seed'):
            if saved.get(key) != job.get(key):
                raise RuntimeError(f'Checkpoint não corresponde ao campo {key}.')
        from src.otimizacao.salas import SolucaoSalas
        resumed = SolucaoSalas(tuple(tuple(pos) for pos in saved['solucao']))
        verificar_salas(inst, resumed, job['k'])
        actual_cost = avaliar_salas(inst, resumed)
        if abs(actual_cost - float(saved['custo'])) > 1e-7:
            raise RuntimeError('Custo do checkpoint não confere com a solução validada.')
        melhor, melhor_custo = resumed, actual_cost
        instante_melhor = float(saved.get('tempo_ate_melhor_s', elapsed_before))
        ultima_melhoria = time.perf_counter()
        validacoes = 1

    def registrar(solucao, fonte):
        nonlocal melhor, melhor_custo, instante_melhor, ultima_melhoria, validacoes
        verificar_salas(inst, solucao, job['k'])
        validacoes += 1
        custo = avaliar_salas(inst, solucao)
        if melhor_custo is None or custo < melhor_custo - TOLERANCE:
            melhor, melhor_custo = solucao, custo
            instante_melhor = elapsed_total()
            ultima_melhoria = time.perf_counter()
            _save(checkpoint, {
                **job, 'status': 'checkpoint_valido', 'custo': melhor_custo,
                'tempo_ate_melhor_s': instante_melhor,
                'tempo_runtime_checkpoint_s': elapsed_total(),
                'solucao': [list(x) for x in melhor.alocacao],
                'validacoes_independentes': validacoes, 'fonte_incumbente': fonte,
            })
            return True
        return False

    ciclo = 0
    if job['method'] == 'regret_tabu':
        while melhor is None and time.perf_counter() < deadline:
            ciclo += 1
            ciclos += 1
            initial = construir(inst, job['k'], job['seed'] + ciclo * 1009,
                                reinicios=1, prazo=deadline)
            construcoes += 1
            if initial.solucao is not None:
                registrar(initial.solucao, 'construtivo')
        if melhor is not None and time.perf_counter() < deadline:
            ciclos += 1
            restante = max(0.0, deadline - time.perf_counter())
            result = _search(inst, melhor, job['k'], job['seed'], restante,
                             STAGNATION_SECONDS)
            if result is not None:
                avaliacoes_tabu += result.avaliacoes
                melhorias_tabu += result.melhorias
                registrar(result.solucao, 'busca_tabu')
                if result.motivo_parada == 'estagnacao':
                    motivo = 'estagnacao_30s'
                elif result.motivo_parada == 'sem_movimentos_admissiveis':
                    motivo = 'vizinhança_esgotada'
    else:
        while time.perf_counter() < deadline:
            agora = time.perf_counter()
            if melhor is not None and ultima_melhoria is not None and \
                    agora - ultima_melhoria >= STAGNATION_SECONDS:
                motivo = 'estagnacao_30s'
                break
            ciclo += 1
            ciclos += 1
            if job['method'] == 'multistart_tabu_room_cycle':
                duracao_fatia = ROOM_CYCLE_CHUNK_SECONDS
            elif job['method'] == 'multistart_tabu_lns':
                duracao_fatia = 15.0
            elif job['method'] == 'multistart_tabu_flow':
                duracao_fatia = FLOW_CYCLE_SECONDS
            elif job['method'] == 'multistart_tabu_lns_flow':
                duracao_fatia = LNS_FLOW_CYCLE_SECONDS
            elif job['method'] == 'multistart_tabu_profile_flow':
                duracao_fatia = PROFILE_FLOW_CYCLE_SECONDS
            elif job['method'] == 'multistart_tabu_profile_flow_lb':
                duracao_fatia = PROFILE_FLOW_GUIDED_CYCLE_SECONDS
            elif job['method'] == 'multistart_tabu_profile_flow_costguided':
                duracao_fatia = PROFILE_FLOW_COST_GUIDED_CYCLE_SECONDS
            else:
                duracao_fatia = CHUNK_SECONDS
            fatia_fim = min(deadline, agora + duracao_fatia)
            constructor = (construir_add if job['method'] == 'multistart_add_tabu'
                           else construir)
            prazo_construtor = (min(fatia_fim, agora + FLOW_CONSTRUCTOR_SECONDS)
                                if job['method'] in ('multistart_tabu_flow',
                                                     'multistart_tabu_lns_flow',
                                                     'multistart_tabu_profile_flow',
                                                     'multistart_tabu_profile_flow_lb',
                                                     'multistart_tabu_profile_flow_costguided')
                                else fatia_fim)
            initial = constructor(inst, job['k'], job['seed'] + ciclo * 1009,
                                  reinicios=1, prazo=prazo_construtor)
            construcoes += 1
            current = initial.solucao if initial.solucao is not None else melhor
            if current is not None:
                registrar(current, 'construtivo')
                restante = max(0.0, fatia_fim - time.perf_counter())
                if restante > 0:
                    if job['method'] == 'multistart_tabu_room_cycle':
                        segundos_tabu = restante * 0.60
                    elif job['method'] == 'multistart_tabu_flow':
                        segundos_tabu = restante * 0.25
                    elif job['method'] == 'multistart_tabu_lns_flow':
                        segundos_tabu = restante * 0.20
                    elif job['method'] == 'multistart_tabu_profile_flow':
                        segundos_tabu = restante * 0.20
                    elif job['method'] == 'multistart_tabu_profile_flow_lb':
                        segundos_tabu = restante * 0.20
                    elif job['method'] == 'multistart_tabu_profile_flow_costguided':
                        segundos_tabu = restante * 0.30
                    elif job['method'] == 'multistart_tabu_lns':
                        segundos_tabu = restante * 0.40
                    elif job['method'] in ('multistart_tabu_multiexchange',
                                            'multistart_tabu_room_exchange'):
                        segundos_tabu = restante * 0.70
                    else:
                        segundos_tabu = restante
                    if ultima_melhoria is None:
                        stall_restante = STAGNATION_SECONDS
                    else:
                        stall_restante = max(1e-6, STAGNATION_SECONDS -
                                             (time.perf_counter() - ultima_melhoria))
                    result = _search(inst, current, job['k'],
                                     job['seed'] + ciclo * 1009, segundos_tabu,
                                     stall_restante)
                    if result is not None:
                        avaliacoes_tabu += result.avaliacoes
                        melhorias_tabu += result.melhorias
                        registrar(result.solucao, 'busca_tabu')
                        current = result.solucao
                    if job['method'] == 'multistart_tabu_multiexchange':
                        restante_exchange = max(0.0, fatia_fim - time.perf_counter())
                        if restante_exchange > 0 and current is not None:
                            exchange = busca_multiexchange(
                                inst, current, job['k'],
                                tempo_limite=restante_exchange)
                            ciclos_exchange += exchange.ciclos
                            movimentos_exchange += exchange.movimentos
                            avaliacoes_exchange += exchange.avaliacoes
                            registrar(exchange.solucao, 'multi_exchange')
                    elif job['method'] == 'multistart_tabu_room_exchange':
                        restante_trocas = max(0.0, fatia_fim - time.perf_counter())
                        if restante_trocas > 0 and current is not None:
                            troca = busca_troca_salas(
                                inst, current, job['k'],
                                tempo_limite=restante_trocas)
                            trocas_salas += troca.trocas
                            pares_salas += troca.pares_avaliados
                            registrar(troca.solucao, 'troca_salas')
                    elif job['method'] == 'multistart_tabu_room_cycle':
                        restante_ciclos = max(0.0, fatia_fim - time.perf_counter())
                        if restante_ciclos > 0 and current is not None:
                            resultado_ciclos = busca_ciclos_salas(
                                inst, current, job['k'],
                                tempo_limite=restante_ciclos)
                            ciclos_quartos += resultado_ciclos.ciclos
                            quartos_movidos += resultado_ciclos.quartos_movidos
                            candidatos_movidos += resultado_ciclos.candidatos_movidos
                            avaliacoes_ciclos += resultado_ciclos.avaliacoes
                            registrar(resultado_ciclos.solucao, 'ciclo_salas')
                    elif job['method'] == 'multistart_tabu_lns':
                        restante_lns = max(0.0, fatia_fim - time.perf_counter())
                        if restante_lns > 0 and current is not None:
                            resultado_lns = busca_lns(
                                inst, current, job['k'],
                                seed=job['seed'] + ciclo * 1009 + 17,
                                tempo_limite=restante_lns,
                                max_iteracoes=1_000_000_000)
                            iteracoes_lns += resultado_lns.iteracoes
                            reparos_lns += resultado_lns.reparos_validos
                            reparos_lns_falhos += resultado_lns.reparos_falhos
                            melhorias_lns += resultado_lns.melhorias
                            registrar(resultado_lns.solucao, 'lns')
                    elif job['method'] == 'multistart_tabu_flow':
                        restante_fluxo = max(0.0, fatia_fim - time.perf_counter())
                        if restante_fluxo > 0 and current is not None:
                            t_fluxo = time.perf_counter()
                            resultado_fluxo = otimizar_perfil(
                                inst, current, job['k'],
                                tempo_limite=restante_fluxo)
                            tempo_fluxo += time.perf_counter() - t_fluxo
                            tipos_otimizados_fluxo += resultado_fluxo.tipos_otimizados
                            aumentos_fluxo += resultado_fluxo.aumentos_fluxo
                            relaxacoes_fluxo += resultado_fluxo.relaxacoes
                            registrar(resultado_fluxo.solucao, 'min_cost_flow_perfil')
                    elif job['method'] == 'multistart_tabu_lns_flow':
                        restante_lns = max(0.0, fatia_fim - time.perf_counter())
                        if restante_lns > 0 and current is not None:
                            resultado_lns = busca_lns(
                                inst, current, job['k'],
                                seed=job['seed'] + ciclo * 1009 + 29,
                                tempo_limite=restante_lns,
                                max_iteracoes=1_000_000_000,
                                otimizar_reparo_fluxo=True)
                            iteracoes_lns += resultado_lns.iteracoes
                            reparos_lns += resultado_lns.reparos_validos
                            reparos_lns_falhos += resultado_lns.reparos_falhos
                            melhorias_lns += resultado_lns.melhorias
                            fluxos_lns += resultado_lns.fluxos_perfil
                            tipos_fluxo_lns += resultado_lns.tipos_otimizados_fluxo
                            aumentos_fluxo_lns += resultado_lns.aumentos_fluxo
                            relaxacoes_fluxo_lns += resultado_lns.relaxacoes_fluxo
                            registrar(resultado_lns.solucao, 'lns_fluxo_perfil')
                    elif job['method'] == 'multistart_tabu_profile_flow':
                        restante_perfil = max(0.0, fatia_fim - time.perf_counter())
                        if restante_perfil > 0 and current is not None:
                            resultado_perfil = busca_perfil_fluxo(
                                inst, current, job['k'],
                                seed=job['seed'] + ciclo * 1009 + 31,
                                tempo_limite=restante_perfil)
                            perfis_propostos += resultado_perfil.propostas
                            perfis_viaveis += resultado_perfil.perfis_viaveis
                            perfis_rejeitados += resultado_perfil.perfis_rejeitados
                            fluxos_perfil += resultado_perfil.fluxos
                            tipos_fluxo_perfil += resultado_perfil.tipos_otimizados
                            aumentos_fluxo_perfil += resultado_perfil.aumentos_fluxo
                            relaxacoes_fluxo_perfil += resultado_perfil.relaxacoes_fluxo
                            registrar(resultado_perfil.solucao, 'perfil_explicito_fluxo')
                    elif job['method'] == 'multistart_tabu_profile_flow_lb':
                        restante_perfil = max(0.0, fatia_fim - time.perf_counter())
                        if restante_perfil > 0 and current is not None:
                            resultado_perfil = busca_perfil_fluxo(
                                inst, current, job['k'],
                                seed=job['seed'] + ciclo * 1009 + 37,
                                tempo_limite=restante_perfil,
                                estrategia='menor_lb',
                                tempo_fluxo_proposta=35.0)
                            perfis_propostos += resultado_perfil.propostas
                            perfis_viaveis += resultado_perfil.perfis_viaveis
                            perfis_rejeitados += resultado_perfil.perfis_rejeitados
                            fluxos_perfil += resultado_perfil.fluxos
                            tipos_fluxo_perfil += resultado_perfil.tipos_otimizados
                            aumentos_fluxo_perfil += resultado_perfil.aumentos_fluxo
                            relaxacoes_fluxo_perfil += resultado_perfil.relaxacoes_fluxo
                            registrar(resultado_perfil.solucao, 'perfil_fluxo_menor_lb')
                    elif job['method'] == 'multistart_tabu_profile_flow_costguided':
                        restante_perfil = max(0.0, fatia_fim - time.perf_counter())
                        if restante_perfil > 0 and current is not None:
                            resultado_perfil = busca_perfil_fluxo(
                                inst, current, job['k'],
                                seed=job['seed'] + ciclo * 1009 + 41,
                                tempo_limite=restante_perfil,
                                estrategia='custo_guiado',
                                tempo_fluxo_proposta=35.0)
                            perfis_propostos += resultado_perfil.propostas
                            perfis_viaveis += resultado_perfil.perfis_viaveis
                            perfis_rejeitados += resultado_perfil.perfis_rejeitados
                            fluxos_perfil += resultado_perfil.fluxos
                            tipos_fluxo_perfil += resultado_perfil.tipos_otimizados
                            aumentos_fluxo_perfil += resultado_perfil.aumentos_fluxo
                            relaxacoes_fluxo_perfil += resultado_perfil.relaxacoes_fluxo
                            registrar(resultado_perfil.solucao,
                                      'perfil_fluxo_custo_guiado')
            if melhor is not None and ultima_melhoria is not None and \
                    time.perf_counter() - ultima_melhoria >= STAGNATION_SECONDS:
                motivo = 'estagnacao_30s'
                break
        else:
            motivo = f'teto_{int(MAX_SECONDS)}s'

    if time.perf_counter() >= deadline:
        motivo = f'teto_{int(MAX_SECONDS)}s'

    if melhor is not None:
        verificar_salas(inst, melhor, job['k'])
    return {
        **job, 'status': 'viavel' if melhor is not None else 'sem_solucao_valida',
        'valida': melhor is not None, 'custo': melhor_custo,
        'tempo_total_s': elapsed_total(),
        'tempo_ate_melhor_s': instante_melhor,
        'motivo_parada': motivo, 'ciclos': ciclos, 'construcoes': construcoes,
        'avaliacoes_tabu': avaliacoes_tabu, 'melhorias_tabu': melhorias_tabu,
        'ciclos_multiexchange': ciclos_exchange,
        'movimentos_multiexchange': movimentos_exchange,
        'avaliacoes_multiexchange': avaliacoes_exchange,
        'trocas_salas': trocas_salas, 'pares_salas_avaliados': pares_salas,
        'ciclos_quartos': ciclos_quartos, 'quartos_movidos': quartos_movidos,
        'candidatos_movidos': candidatos_movidos,
        'avaliacoes_ciclos_salas': avaliacoes_ciclos,
        'iteracoes_lns': iteracoes_lns, 'reparos_lns': reparos_lns,
        'reparos_lns_falhos': reparos_lns_falhos, 'melhorias_lns': melhorias_lns,
        'tempo_fluxo_s': tempo_fluxo,
        'tipos_otimizados_fluxo': tipos_otimizados_fluxo,
        'aumentos_fluxo': aumentos_fluxo, 'relaxacoes_fluxo': relaxacoes_fluxo,
        'fluxos_lns': fluxos_lns, 'tipos_fluxo_lns': tipos_fluxo_lns,
        'aumentos_fluxo_lns': aumentos_fluxo_lns,
        'relaxacoes_fluxo_lns': relaxacoes_fluxo_lns,
        'perfis_propostos': perfis_propostos, 'perfis_viaveis': perfis_viaveis,
        'perfis_rejeitados': perfis_rejeitados, 'fluxos_perfil': fluxos_perfil,
        'tipos_fluxo_perfil': tipos_fluxo_perfil,
        'aumentos_fluxo_perfil': aumentos_fluxo_perfil,
        'relaxacoes_fluxo_perfil': relaxacoes_fluxo_perfil,
        'validacoes_independentes': validacoes, 'unidade': inst.unidade_distancias,
        'limite_estagnacao_s': STAGNATION_SECONDS, 'limite_maximo_s': MAX_SECONDS,
        'solucao': [list(x) for x in melhor.alocacao] if melhor is not None else None,
    }


def _worker(job_path):
    job = json.loads(Path(job_path).read_text(encoding='utf-8'))
    result = execute(job)
    _save(Path(job['output']), result)
    print(json.dumps({'instance': job['instance'], 'K': job['k'],
                      'method': job['method'], 'seed': job['seed'],
                      'valid': result['valida'], 'stop': result['motivo_parada']}))


def _summarize(out):
    results = [json.loads(p.read_text(encoding='utf-8'))
               for p in sorted(out.glob('run_*.json'))]
    fields = ['instance', 'k', 'method', 'seed', 'status', 'valida', 'custo',
              'tempo_total_s', 'tempo_ate_melhor_s', 'motivo_parada', 'ciclos',
              'construcoes', 'avaliacoes_tabu', 'melhorias_tabu',
              'ciclos_multiexchange', 'movimentos_multiexchange',
              'avaliacoes_multiexchange', 'validacoes_independentes', 'unidade']
    temp = out / 'execucoes.csv.tmp'
    with temp.open('w', encoding='utf-8', newline='') as arquivo:
        writer = csv.DictWriter(arquivo, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: registro.get(key) for key in fields}
                         for registro in results)
    os.replace(temp, out / 'execucoes.csv')


def main():
    global INSTANCES, SEEDS
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--saida')
    parser.add_argument('--metodo', choices=METHODS)
    parser.add_argument('--todas-instancias', action='store_true',
                        help='Executa em todos os .txt legíveis de data/instancias.')
    parser.add_argument('--seed-inicial', type=int, default=42)
    parser.add_argument('--quantidade-seeds', type=int, default=5)
    parser.add_argument('--resume-from', help='Pasta de checkpoints validados para retomar execuções interrompidas.')
    parser.add_argument('--worker-job', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.workers < 1:
        raise SystemExit('--workers deve ser positivo.')
    if args.worker_job:
        _worker(args.worker_job)
        return

    if args.todas_instancias:
        arquivos = sorted((ROOT / 'data/instancias').glob('*.txt'),
                          key=lambda p: p.name.casefold())
        invalidas = []
        for caminho in arquivos:
            try:
                ler_instancia(caminho)
            except Exception as exc:
                invalidas.append(f'{caminho.name}: {exc}')
        if invalidas:
            detalhe = '\n'.join(invalidas)
            raise SystemExit('Há arquivos .txt que não podem ser lidos; nenhuma execução foi iniciada:\n' + detalhe)
        INSTANCES = tuple(caminho.name for caminho in arquivos)
        if not INSTANCES:
            raise SystemExit('Nenhum arquivo .txt foi encontrado em data/instancias.')
    if args.quantidade_seeds < 1:
        raise SystemExit('--quantidade-seeds deve ser positiva.')
    SEEDS = tuple(range(args.seed_inicial,
                        args.seed_inicial + args.quantidade_seeds))

    out = ROOT / (args.saida or 'resultados/lote_multiexchange')
    out.mkdir(parents=True, exist_ok=True)
    resume_dir = None
    if args.resume_from:
        resume_dir = Path(args.resume_from)
        resume_dir = resume_dir if resume_dir.is_absolute() else ROOT / resume_dir
        if not resume_dir.is_dir():
            raise SystemExit(f'Pasta de checkpoints não existe: {resume_dir}')
    selected_methods = (args.metodo,) if args.metodo else METHODS
    instance_hashes = {name: sha256(ROOT / 'data/instancias' / name) for name in INSTANCES}
    sources = [ROOT / 'ferramentas/experimento_parada_adaptativa.py',
               ROOT / 'src/otimizacao/busca_tabu.py',
               ROOT / 'src/otimizacao/busca_multiexchange.py',
               ROOT / 'src/otimizacao/busca_troca_salas.py',
               ROOT / 'src/otimizacao/busca_ciclos_salas.py',
               ROOT / 'src/otimizacao/busca_lns.py',
               ROOT / 'src/otimizacao/busca_perfil_fluxo.py',
               ROOT / 'src/otimizacao/fluxo_perfil.py',
               ROOT / 'src/otimizacao/construtivo.py',
               ROOT / 'src/otimizacao/construtivo_add.py',
               ROOT / 'src/otimizacao/salas.py']
    source_hashes = {str(p.relative_to(ROOT)): sha256(p) for p in sources}
    old_config = None
    if resume_dir is not None:
        old_config_path = resume_dir / 'config.json'
        if not old_config_path.exists():
            raise SystemExit('A pasta de retomada não tem config.json para conferir as entradas.')
        old_config = json.loads(old_config_path.read_text(encoding='utf-8'))
        if old_config.get('instance_sha256') != instance_hashes:
            raise SystemExit('As instâncias diferem dos checkpoints.')
        if old_config.get('types_per_school') != [1, 2]:
            raise SystemExit('O limite de tipos dos checkpoints não corresponde a K=1/2.')
    config = {'workers': args.workers, 'instances': list(INSTANCES), 'methods': list(selected_methods),
              'seeds': list(SEEDS), 'stagnation_s': STAGNATION_SECONDS,
              'hard_timeout_s': MAX_SECONDS, 'chunk_s': CHUNK_SECONDS,
              'room_cycle_chunk_s': ROOM_CYCLE_CHUNK_SECONDS,
              'lns_chunk_s': 15.0,
              'lns_flow_chunk_s': 45.0,
              'profile_flow_cycle_s': PROFILE_FLOW_CYCLE_SECONDS,
              'profile_flow_guided_cycle_s': PROFILE_FLOW_GUIDED_CYCLE_SECONDS,
              'profile_flow_cost_guided_cycle_s': PROFILE_FLOW_COST_GUIDED_CYCLE_SECONDS,
              'flow_cycle_s': FLOW_CYCLE_SECONDS,
              'flow_constructor_s': FLOW_CONSTRUCTOR_SECONDS,
              'resume_from': str(resume_dir) if resume_dir else None,
              'resume_source_sha256': old_config.get('source_sha256') if old_config else None,
              'types_per_school': [1, 2], 'max_tabu_evaluations_per_call': MAX_EVALS,
              'instance_sha256': instance_hashes,
              'source_sha256': source_hashes, 'python': platform.python_version(),
              'backend': 'NumPy CPU'}
    config_path = out / 'config.json'
    if config_path.exists() and json.loads(config_path.read_text(encoding='utf-8')) != config:
        raise SystemExit('A pasta contém outra configuração; escolha novo destino.')
    if not config_path.exists():
        _save(config_path, config)
    _summarize(out)

    job_dir = out / 'jobs'
    job_dir.mkdir(exist_ok=True)
    jobs = []
    for instance in INSTANCES:
        for k in (1, 2):
            for method in selected_methods:
                for seed in SEEDS:
                    name = f'{instance}_{k}_{method}_{seed}'
                    output = out / f'run_{name}.json'
                    if output.exists():
                        continue
                    expected = {'instance': instance, 'instance_sha256': instance_hashes[instance],
                                'k': k,
                                'method': method, 'seed': seed}
                    if resume_dir is not None:
                        source_run = resume_dir / output.name
                        if source_run.exists():
                            saved_run = json.loads(source_run.read_text(encoding='utf-8'))
                            if any(saved_run.get(key) != value for key, value in expected.items()):
                                raise SystemExit(f'Resultado incompatível: {source_run.name}')
                            if saved_run.get('solucao') is not None:
                                from src.otimizacao.salas import SolucaoSalas
                                check_instance = ler_instancia(ROOT / 'data/instancias' / instance)
                                check_solution = SolucaoSalas(tuple(tuple(p) for p in saved_run['solucao']))
                                verificar_salas(check_instance, check_solution, k)
                                saved_run['custo'] = avaliar_salas(check_instance, check_solution)
                            saved_run['retomado_final_de'] = str(source_run)
                            _save(output, saved_run)
                            continue
                    prior_elapsed = 0.0
                    remaining = MAX_SECONDS
                    target_checkpoint = output.with_suffix('.checkpoint')
                    if resume_dir is not None:
                        source_checkpoint = resume_dir / target_checkpoint.name
                        if source_checkpoint.exists():
                            saved = json.loads(source_checkpoint.read_text(encoding='utf-8'))
                            if any(saved.get(key) != value for key, value in expected.items()):
                                raise SystemExit(f'Checkpoint incompatível: {source_checkpoint.name}')
                            from src.otimizacao.salas import SolucaoSalas
                            check_instance = ler_instancia(ROOT / 'data/instancias' / instance)
                            check_solution = SolucaoSalas(tuple(tuple(p) for p in saved['solucao']))
                            verificar_salas(check_instance, check_solution, k)
                            check_cost = avaliar_salas(check_instance, check_solution)
                            if abs(check_cost - float(saved['custo'])) > 1e-7:
                                raise SystemExit(f'Custo inconsistente no checkpoint: {source_checkpoint.name}')
                            _save(target_checkpoint, saved)
                            elapsed_saved = float(saved.get(
                                'tempo_runtime_checkpoint_s',
                                saved.get('tempo_ate_melhor_s', 0.0)))
                            # Reserve a full patience window for possible work
                            # after the last saved incumbent.
                            prior_elapsed = min(MAX_SECONDS, elapsed_saved + STAGNATION_SECONDS)
                            remaining = max(0.0, MAX_SECONDS - prior_elapsed)
                            if remaining <= 0:
                                saved.update(status=f'teto_{int(MAX_SECONDS)}s_checkpoint',
                                             valida=True, tempo_total_s=MAX_SECONDS,
                                             motivo_parada=f'teto_{int(MAX_SECONDS)}s_checkpoint')
                                _save(output, saved)
                                continue
                    job = {'instance': instance, 'instance_sha256': instance_hashes[instance],
                           'k': k, 'method': method,
                           'seed': seed, 'output': str(output),
                           'resume_prior_elapsed_s': prior_elapsed,
                           'remaining_budget_s': remaining}
                    job_path = job_dir / f'{name}.json'
                    _save(job_path, job)
                    jobs.append(job_path)

    def launch(job_path):
        job = json.loads(job_path.read_text(encoding='utf-8'))
        start = time.perf_counter()
        try:
            proc = subprocess.run([sys.executable, str(Path(__file__).resolve()),
                                   '--worker-job', str(job_path)], cwd=ROOT,
                                  capture_output=True, text=True,
                                  timeout=float(job.get('remaining_budget_s', MAX_SECONDS)))
            if proc.returncode:
                raise RuntimeError(proc.stderr[-2000:])
            result = json.loads(Path(job['output']).read_text(encoding='utf-8'))
        except subprocess.TimeoutExpired:
            result_path = Path(job['output'])
            checkpoint = result_path.with_suffix('.checkpoint')
            if checkpoint.exists():
                result = json.loads(checkpoint.read_text(encoding='utf-8'))
                result.update(status=f'teto_{int(MAX_SECONDS)}s_checkpoint', valida=True,
                              tempo_total_s=MAX_SECONDS,
                              motivo_parada=f'teto_{int(MAX_SECONDS)}s_supervisor')
            else:
                result = {**job, 'status': f'sem_solucao_ate_{int(MAX_SECONDS)}s', 'valida': False,
                          'custo': None, 'tempo_total_s': MAX_SECONDS,
                          'tempo_ate_melhor_s': None,
                          'motivo_parada': f'teto_{int(MAX_SECONDS)}s_supervisor',
                          'solucao': None}
            _save(Path(job['output']), result)
        result['tempo_supervisor_s'] = time.perf_counter() - start
        # Recheck the reported incumbent in the parent, independently of worker state.
        if result.get('solucao') is not None:
            inst = ler_instancia(ROOT / 'data/instancias' / job['instance'])
            from src.otimizacao.salas import SolucaoSalas
            solution = SolucaoSalas(tuple(tuple(pos) for pos in result['solucao']))
            verificar_salas(inst, solution, job['k'])
            result['custo'] = avaliar_salas(inst, solution)
        _save(Path(job['output']), result)
        return result

    total = len(jobs)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(launch, path): path for path in jobs}
        for done, future in enumerate(as_completed(futures), 1):
            result = future.result()
            if done % 10 == 0 or done == total:
                _summarize(out)
            print(f"{done}/{total}: {result['instance']} K={result['k']} "
                  f"{result['method']} seed={result['seed']} "
                  f"válida={result.get('valida')} tempo={result.get('tempo_total_s'):.1f}s "
                  f"parada={result.get('motivo_parada')}", flush=True)
    _summarize(out)
    print(f'CONCLUÍDO {out.relative_to(ROOT)}')


if __name__ == '__main__':
    main()
