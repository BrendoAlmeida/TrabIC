"""Busca de vizinhança grande com destruição e reconstrução por regret.

Este módulo é experimental. Cada reparo mantém os candidatos fora da vizinhança
fixos, reatribui os removidos e só devolve soluções aprovadas pelo verificador
independente. A busca não certifica ótimo local ou global.
"""
from dataclasses import dataclass
import math
import random
import time
from numbers import Integral

from .fluxo_perfil import otimizar_perfil
from .salas import SolucaoSalas, avaliar_salas, verificar_salas


@dataclass(frozen=True)
class ResultadoLNS:
    solucao: SolucaoSalas
    custo_inicial: float
    custo_final: float
    iteracoes: int
    reparos_validos: int
    reparos_falhos: int
    aceitas: int
    melhorias: int
    motivo_parada: str
    fluxos_perfil: int
    tipos_otimizados_fluxo: int
    aumentos_fluxo: int
    relaxacoes_fluxo: int


def _estado(instancia, alocacao):
    tipos = [c.tipo_prova for c in instancia.candidatos]
    membros = [[set() for _ in e.capacidades_salas] for e in instancia.escolas]
    tipo_sala = [[None for _ in e.capacidades_salas] for e in instancia.escolas]
    tipos_escola = [set() for _ in instancia.escolas]
    for i, (j, r) in enumerate(alocacao):
        if j < 0:
            continue
        membros[j][r].add(i)
        tipo_sala[j][r] = tipos[i]
        tipos_escola[j].add(tipos[i])
    return tipos, membros, tipo_sala, tipos_escola


def _destinos(instancia, i, tipos_por_local, membros, tipo_sala, tipos_escola):
    """Um destino por escola, usando melhor encaixe de sala disponível."""
    tipo = instancia.candidatos[i].tipo_prova
    destinos = []
    for j, escola in enumerate(instancia.escolas):
        if tipo not in tipos_escola[j] and len(tipos_escola[j]) >= tipos_por_local:
            continue
        salas = [r for r, capacidade in enumerate(escola.capacidades_salas)
                 if len(membros[j][r]) < capacidade and tipo_sala[j][r] in (None, tipo)]
        if not salas:
            continue
        r = min(salas, key=lambda sala: (
            escola.capacidades_salas[sala] - len(membros[j][sala]), sala))
        destinos.append((float(instancia.matriz_distancias[i][j]), j, r))
    destinos.sort(key=lambda x: (x[0], x[1], x[2]))
    return destinos


def _reparar(instancia, base, removidos, tipos_por_local, rng, alpha, deadline):
    """Reconstrói uma vizinhança parcialmente fixa por regret de inserção."""
    tipos, membros, tipo_sala, tipos_escola = _estado(instancia, base)
    pendentes = set(removidos)
    insercoes = 0
    while pendentes:
        if deadline is not None and time.perf_counter() >= deadline:
            return None, insercoes
        alternativas = {}
        for i in pendentes:
            opcoes = _destinos(instancia, i, tipos_por_local, membros,
                               tipo_sala, tipos_escola)
            if not opcoes:
                return None, insercoes
            alternativas[i] = opcoes

        forcados = [i for i, opcoes in alternativas.items() if len(opcoes) == 1]
        if forcados:
            i = rng.choice(forcados)
            _, j, r = alternativas[i][0]
            tipo = tipos[i]
            membros[j][r].add(i)
            tipo_sala[j][r] = tipo
            tipos_escola[j].add(tipo)
            base[i] = (j, r)
            pendentes.remove(i)
            insercoes += 1
            continue

        regrets = []
        for i, opcoes in alternativas.items():
            regret = math.inf if len(opcoes) == 1 else opcoes[1][0] - opcoes[0][0]
            regrets.append((regret, i))
        regrets.sort(key=lambda x: (-x[0], x[1]))
        amplitude = regrets[0][0] - regrets[-1][0]
        limite = regrets[0][0] - alpha * amplitude
        rcl = [i for regret, i in regrets if regret >= limite]
        i = rcl[rng.randrange(len(rcl))]
        _, j, r = alternativas[i][0]
        tipo = tipos[i]
        membros[j][r].add(i)
        tipo_sala[j][r] = tipo
        tipos_escola[j].add(tipo)
        base[i] = (j, r)
        pendentes.remove(i)
        insercoes += 1

    solucao = SolucaoSalas(tuple(base))
    return solucao, insercoes


def _ruinar(instancia, atual, rng, modo):
    alocacao = list(atual.alocacao)
    n = len(alocacao)
    if not n:
        return alocacao, []
    ocupadas = {}
    for i, (j, _) in enumerate(alocacao):
        ocupadas.setdefault(j, []).append(i)
    escolas = list(ocupadas)

    if modo == "escolas_aleatorias":
        quantidade = min(len(escolas), rng.choice((1, 1, 2, 2, 3)))
        selecionadas = rng.sample(escolas, quantidade)
        removidos = [i for j in selecionadas for i in ocupadas[j]]
    else:
        # Remove primeiro os candidatos mais caros acima do seu limite inferior
        # individual; dispersa o conjunto para mudar mais de uma região.
        excessos = [float(instancia.matriz_distancias[i][alocacao[i][0]]) -
                    min(float(x) for x in instancia.matriz_distancias[i])
                    for i in range(n)]
        quantidade = max(1, min(n, int(math.ceil(n * rng.uniform(0.08, 0.24)))))
        ordem = sorted(range(n), key=lambda i: (-excessos[i], i))
        candidatos_top = ordem[:max(quantidade, min(n, quantidade * 3))]
        removidos = rng.sample(candidatos_top, quantidade)

    for i in removidos:
        alocacao[i] = (-1, -1)
    return alocacao, removidos


def busca_lns(instancia, solucao, tipos_por_local, seed=0, *, tempo_limite=None,
              max_iteracoes=100_000, alpha=0.2, temperatura_inicial=0.02,
              resfriamento=0.995, otimizar_reparo_fluxo=False):
    """Destrói e reconstrói subconjuntos; mantém o melhor estado válido.

    Alterna remoção de uma a três escolas e remoção de candidatos de maior
    excesso de distância. Reparos usam regret aleatorizado, capacidade e tipos
    por sala. Uma aceitação annealing limitada diversifica o estado atual; a
    solução devolvida continua sendo sempre a melhor encontrada.
    """
    if not isinstance(solucao, SolucaoSalas):
        raise ValueError("solucao deve ser SolucaoSalas")
    if isinstance(tipos_por_local, bool) or not isinstance(tipos_por_local, Integral) or tipos_por_local < 1:
        raise ValueError("tipos_por_local deve ser inteiro positivo")
    if isinstance(max_iteracoes, bool) or not isinstance(max_iteracoes, Integral) or max_iteracoes < 0:
        raise ValueError("max_iteracoes deve ser inteiro não negativo")
    if not isinstance(otimizar_reparo_fluxo, bool):
        raise ValueError('otimizar_reparo_fluxo deve ser booleano')
    if otimizar_reparo_fluxo and tempo_limite is None:
        raise ValueError('reparo por fluxo exige tempo_limite finito')
    if tempo_limite is not None and (not math.isfinite(tempo_limite) or tempo_limite < 0):
        raise ValueError("tempo_limite deve ser finito e não negativo")
    if not 0 <= alpha <= 1:
        raise ValueError("alpha deve estar entre 0 e 1")
    if not math.isfinite(temperatura_inicial) or temperatura_inicial < 0:
        raise ValueError("temperatura_inicial deve ser finita e não negativa")
    if not math.isfinite(resfriamento) or not 0 < resfriamento <= 1:
        raise ValueError("resfriamento deve estar no intervalo (0, 1]")

    verificar_salas(instancia, solucao, tipos_por_local)
    inicio = time.perf_counter()
    deadline = None if tempo_limite is None else inicio + tempo_limite
    atual = solucao
    melhor = solucao
    custo_atual = avaliar_salas(instancia, atual)
    custo_melhor = custo_atual
    rng = random.Random(seed)
    iteracoes = reparos_validos = reparos_falhos = aceitas = melhorias = 0
    fluxos_perfil = tipos_otimizados_fluxo = aumentos_fluxo = relaxacoes_fluxo = 0
    motivo = "vizinhança_esgotada"

    while iteracoes < max_iteracoes:
        if deadline is not None and time.perf_counter() >= deadline:
            motivo = "limite_tempo"
            break
        modo = "escolas_aleatorias" if iteracoes % 2 == 0 else "candidatos_caros"
        base, removidos = _ruinar(instancia, atual, rng, modo)
        if not removidos:
            motivo = "sem_candidatos"
            break
        proposta, _ = _reparar(instancia, base, removidos,
                               tipos_por_local, rng, alpha, deadline)
        iteracoes += 1
        if proposta is None:
            reparos_falhos += 1
            continue

        if otimizar_reparo_fluxo:
            restante = max(0.0, deadline - time.perf_counter())
            if restante > 0:
                fluxo = otimizar_perfil(instancia, proposta, tipos_por_local,
                                        tempo_limite=restante)
                proposta = fluxo.solucao
                fluxos_perfil += 1
                tipos_otimizados_fluxo += fluxo.tipos_otimizados
                aumentos_fluxo += fluxo.aumentos_fluxo
                relaxacoes_fluxo += fluxo.relaxacoes

        verificar_salas(instancia, proposta, tipos_por_local)
        reparos_validos += 1
        custo_proposta = avaliar_salas(instancia, proposta)
        if custo_proposta < custo_melhor - 1e-10:
            melhor = proposta
            custo_melhor = custo_proposta
            melhorias += 1

        delta = custo_proposta - custo_atual
        temperatura = temperatura_inicial * max(1.0, custo_atual) * (resfriamento ** iteracoes)
        if delta <= 1e-10 or (temperatura > 0 and rng.random() < math.exp(-delta / temperatura)):
            atual = proposta
            custo_atual = custo_proposta
            aceitas += 1

    else:
        motivo = "limite_iteracoes"

    verificar_salas(instancia, melhor, tipos_por_local)
    return ResultadoLNS(melhor, avaliar_salas(instancia, solucao), custo_melhor,
                        iteracoes, reparos_validos, reparos_falhos, aceitas,
                        melhorias, motivo, fluxos_perfil,
                        tipos_otimizados_fluxo, aumentos_fluxo, relaxacoes_fluxo)
