"""Busca LNS sobre perfis de tipos avaliada por min-cost flow.

Cada candidato de perfil muda tipos de quartos (abrindo capacidade em escolas
ou trocando tipo de quartos). Um min-cost flow resolve exatamente a atribuição
de candidatos para esse perfil. A busca mantém o melhor perfil viável e usa
annealing limitado apenas para diversificar perfis intermediários.
"""
from dataclasses import dataclass
from itertools import combinations
import math
import random
import time
from numbers import Integral

import numpy as np

from .fluxo_perfil import otimizar_perfil_explicito
from .salas import SolucaoSalas, avaliar_salas, verificar_salas


@dataclass(frozen=True)
class ResultadoBuscaPerfil:
    solucao: SolucaoSalas
    custo_inicial: float
    custo_final: float
    propostas: int
    perfis_viaveis: int
    perfis_rejeitados: int
    aceitos: int
    melhorias: int
    fluxos: int
    tipos_otimizados: int
    aumentos_fluxo: int
    relaxacoes_fluxo: int
    motivo_parada: str


def _perfil_inicial(instancia, solucao, tipos_por_local, rng):
    perfil = [[None] * len(e.capacidades_salas) for e in instancia.escolas]
    for i, (j, r) in enumerate(solucao.alocacao):
        perfil[j][r] = instancia.candidatos[i].tipo_prova
    demandas = {t: sum(c.tipo_prova == t for c in instancia.candidatos)
                for t in {c.tipo_prova for c in instancia.candidatos}}
    for j, escola in enumerate(instancia.escolas):
        ativos = {t for t in perfil[j] if t is not None}
        if not ativos:
            continue
        if tipos_por_local == 1:
            tipo = next(iter(ativos))
            for r, cap in enumerate(escola.capacidades_salas):
                if cap > 0:
                    perfil[j][r] = tipo
        else:
            falta = {}
            for t in ativos:
                cap_t = sum(escola.capacidades_salas[r]
                            for r, room_type in enumerate(perfil[j]) if room_type == t)
                falta[t] = demandas[t] - cap_t
            for r, cap in enumerate(escola.capacidades_salas):
                if cap <= 0 or perfil[j][r] is not None:
                    continue
                perfil[j][r] = max(sorted(ativos), key=lambda t: (falta[t], rng.random()))
                falta[perfil[j][r]] -= cap
    return perfil


def _tipos_escola(perfil):
    return [set(t for t in linha if t is not None) for linha in perfil]


def _prioridades_escolas(instancia, solucao, matriz=None):
    """Prioriza escolas de origem e alternativas para alunos caros.

    Para cada candidato, soma a economia potencial entre sua escola atual e a
    melhor escola alternativa, sem impor o perfil atual de tipos. Essa relaxação
    serve apenas para ordenar movimentos; a viabilidade e o custo são decididos
    pelo gerador de perfis e pelo fluxo exato.
    """
    if matriz is None:
        matriz = np.asarray(instancia.matriz_distancias, dtype=float)
    quantidade = len(instancia.escolas)
    prioridades = [0.0] * quantidade
    if quantidade < 2:
        return prioridades
    for i, (atual, _) in enumerate(solucao.alocacao):
        custos = matriz[i]
        alternativa, destino = min(
            (float(custos[j]), j) for j in range(quantidade) if j != atual)
        economia = max(0.0, float(custos[atual]) - alternativa)
        if economia:
            prioridades[atual] += economia
            prioridades[destino] += economia
    return prioridades


def _mutar(instancia, perfil, tipos_por_local, tipos, rng):
    proposta = [linha.copy() for linha in perfil]
    capacidades = [e.capacidades_salas for e in instancia.escolas]
    ativos = [j for j, linha in enumerate(perfil) if any(t is not None for t in linha)]
    vazias = [j for j, linha in enumerate(perfil) if not any(t is not None for t in linha)]

    if tipos_por_local == 1:
        if vazias and ativos and rng.random() < 0.75:
            j = rng.choice(vazias)
            fonte = rng.choice(ativos)
            t = next(x for x in perfil[fonte] if x is not None)
            for r, cap in enumerate(capacidades[j]):
                if cap > 0:
                    proposta[j][r] = t
            return proposta
        if len(ativos) >= 3 and rng.random() < 0.45:
            j, k, ell = rng.sample(ativos, 3)
            tj = next(x for x in perfil[j] if x is not None)
            tk = next(x for x in perfil[k] if x is not None)
            tell = next(x for x in perfil[ell] if x is not None)
            if len({tj, tk, tell}) > 1:
                # Rotate three school-wide types in one profile move.
                for r, cap in enumerate(capacidades[j]):
                    if cap > 0:
                        proposta[j][r] = tk
                for r, cap in enumerate(capacidades[k]):
                    if cap > 0:
                        proposta[k][r] = tell
                for r, cap in enumerate(capacidades[ell]):
                    if cap > 0:
                        proposta[ell][r] = tj
                return proposta
        if len(ativos) >= 2:
            pares = []
            for p, j in enumerate(ativos):
                for k in ativos[p + 1:]:
                    tj = next(x for x in perfil[j] if x is not None)
                    tk = next(x for x in perfil[k] if x is not None)
                    cap_j = sum(capacidades[j])
                    cap_k = sum(capacidades[k])
                    if tj != tk and cap_j == cap_k:
                        pares.append((j, k, tj, tk))
            if pares:
                j, k, tj, tk = rng.choice(pares)
                for r, cap in enumerate(capacidades[j]):
                    if cap > 0:
                        proposta[j][r] = tk
                for r, cap in enumerate(capacidades[k]):
                    if cap > 0:
                        proposta[k][r] = tj
                return proposta
            if rng.random() < 0.5:
                j, k = rng.sample(ativos, 2)
                tj = next(x for x in perfil[j] if x is not None)
                tk = next(x for x in perfil[k] if x is not None)
                if tj != tk:
                    for r, cap in enumerate(capacidades[j]):
                        if cap > 0:
                            proposta[j][r] = tk
                    for r, cap in enumerate(capacidades[k]):
                        if cap > 0:
                            proposta[k][r] = tj
                    return proposta
        if ativos:
            j = rng.choice(ativos)
            old = next(x for x in perfil[j] if x is not None)
            novos = [t for t in tipos if t != old]
            if novos:
                novo = rng.choice(novos)
                for r, cap in enumerate(capacidades[j]):
                    if cap > 0:
                        proposta[j][r] = novo
                return proposta
        return None

    # Para K>=2, tente abrir/recolorir quartos ou girar tipos entre três escolas.
    # A verificação da capacidade agregada por tipo vem depois da mutação.
    if vazias and rng.random() < 0.50:
        j = rng.choice(vazias)
        escolha = rng.sample(list(tipos), min(tipos_por_local, len(tipos)))
        quartos = [r for r, cap in enumerate(capacidades[j]) if cap > 0]
        if not quartos:
            return None
        rng.shuffle(quartos)
        for pos, r in enumerate(quartos):
            proposta[j][r] = escolha[min(pos, len(escolha) - 1)]
        return proposta

    if rng.random() < 0.25:
        escolas_com_quartos = [j for j, e in enumerate(instancia.escolas)
                               if any(cap > 0 for cap in e.capacidades_salas)]
        if len(escolas_com_quartos) >= 3:
            escolas = rng.sample(escolas_com_quartos, 3)
            selecionadas = []
            for j in escolas:
                quartos = [r for r, cap in enumerate(capacidades[j]) if cap > 0]
                selecionadas.append((j, rng.choice(quartos)))
            etiquetas = [perfil[j][r] for j, r in selecionadas]
            if len(set(etiquetas)) > 1:
                for pos, (j, r) in enumerate(selecionadas):
                    proposta[j][r] = etiquetas[(pos + 1) % 3]
                if all(len({t for t in linha if t is not None}) <= tipos_por_local
                       for linha in proposta):
                    return proposta

    if rng.random() < 0.75:
        quartos = [(j, r) for j, e in enumerate(instancia.escolas)
                   for r, cap in enumerate(e.capacidades_salas) if cap > 0]
        pares_equivalentes = [(j, r, k, s) for pos, (j, r) in enumerate(quartos)
                              for k, s in quartos[pos + 1:]
                              if j != k and capacidades[j][r] == capacidades[k][s]
                              and perfil[j][r] != perfil[k][s]]
        if pares_equivalentes:
            j, r, k, s = rng.choice(pares_equivalentes)
            proposta[j][r], proposta[k][s] = proposta[k][s], proposta[j][r]
        elif len(quartos) >= 2:
            j, r = rng.choice(quartos)
            restantes = [(k, s) for k, s in quartos if k != j]
            if restantes:
                k, s = rng.choice(restantes)
                proposta[j][r], proposta[k][s] = proposta[k][s], proposta[j][r]
    else:
        elegiveis = [(j, r) for j, e in enumerate(instancia.escolas)
                     for r, cap in enumerate(e.capacidades_salas) if cap > 0]
        if elegiveis:
            j, r = rng.choice(elegiveis)
            ativos_j = _tipos_escola(perfil)[j]
            opcoes = [None] + [t for t in tipos if t != perfil[j][r]]
            rng.shuffle(opcoes)
            for novo in opcoes:
                tentativa = proposta[j].copy()
                tentativa[r] = novo
                if len({t for t in tentativa if t is not None}) <= tipos_por_local:
                    proposta[j] = tentativa
                    break
    if proposta == perfil:
        return None
    if any(len({t for t in linha if t is not None}) > tipos_por_local
           for linha in proposta):
        return None
    return proposta


def _vizinhos_lb(instancia, perfil, tipos_por_local, tipos, rng,
                  candidatos_tipo, matriz, deadline, limite_swaps=300,
                  limite_avaliacoes=250, prioridades_escolas=None):
    """Gera perfis factíveis por capacidade e os ordena por limite inferior."""
    capacidades = [e.capacidades_salas for e in instancia.escolas]
    demandas = {t: len(candidatos_tipo[t]) for t in tipos}
    candidatos = {}
    avaliados = 0
    interrompido = False

    def adicionar(proposta):
        nonlocal avaliados, interrompido
        if interrompido:
            return
        if time.perf_counter() >= deadline or avaliados >= limite_avaliacoes:
            interrompido = True
            return
        avaliados += 1
        if proposta == perfil or any(len({t for t in linha if t is not None}) > tipos_por_local
                                     for linha in proposta):
            return
        capacidade_tipo = {t: 0 for t in tipos}
        for j, linha in enumerate(proposta):
            for r, t in enumerate(linha):
                if t is not None:
                    capacidade_tipo[t] += capacidades[j][r]
        if any(capacidade_tipo[t] < demandas[t] for t in tipos):
            return
        score = 0.0
        for t in tipos:
            idx = candidatos_tipo[t]
            schools = [j for j, linha in enumerate(proposta)
                       if any(x == t for x in linha)]
            if len(idx) and not schools:
                return
            if len(idx):
                score += float(np.min(matriz[np.ix_(idx, schools)], axis=1).sum())
        key = tuple(tuple(line) for line in proposta)
        candidatos[key] = score

    if prioridades_escolas is None:
        ordem_escolas = list(range(len(instancia.escolas)))
    else:
        ordem_escolas = sorted(range(len(instancia.escolas)),
                               key=lambda j: (-prioridades_escolas[j], j))

    # K=1: recolor uma escola inteira para outro tipo.
    if tipos_por_local == 1:
        for j in ordem_escolas:
            escola = instancia.escolas[j]
            if interrompido:
                break
            atual = next((t for t in perfil[j] if t is not None), None)
            for tipo in tipos:
                if interrompido:
                    break
                if tipo == atual:
                    continue
                proposta = [line.copy() for line in perfil]
                proposta[j] = [tipo if cap > 0 else None
                               for cap in escola.capacidades_salas]
                adicionar(proposta)
        ativos = [j for j in ordem_escolas
                  if any(t is not None for t in perfil[j])]
        for pos, j in enumerate(ativos):
            if interrompido:
                break
            tj = next(t for t in perfil[j] if t is not None)
            for k in ativos[pos + 1:]:
                if interrompido:
                    break
                tk = next(t for t in perfil[k] if t is not None)
                if tj == tk:
                    continue
                proposta = [line.copy() for line in perfil]
                proposta[j] = [tk if cap > 0 else None
                               for cap in capacidades[j]]
                proposta[k] = [tj if cap > 0 else None
                               for cap in capacidades[k]]
                adicionar(proposta)
        for j, k, ell in combinations(ativos, 3):
            if interrompido:
                break
            tj = next(t for t in perfil[j] if t is not None)
            tk = next(t for t in perfil[k] if t is not None)
            tell = next(t for t in perfil[ell] if t is not None)
            if len({tj, tk, tell}) < 2:
                continue
            for new_j, new_k, new_ell in ((tk, tell, tj), (tell, tj, tk)):
                proposta = [line.copy() for line in perfil]
                proposta[j] = [new_j if cap > 0 else None
                               for cap in capacidades[j]]
                proposta[k] = [new_k if cap > 0 else None
                               for cap in capacidades[k]]
                proposta[ell] = [new_ell if cap > 0 else None
                                  for cap in capacidades[ell]]
                adicionar(proposta)
    else:
        # Recolor um quarto ou trocar rótulos entre quartos de igual capacidade.
        for j in ordem_escolas:
            escola = instancia.escolas[j]
            if interrompido:
                break
            for r, cap in enumerate(escola.capacidades_salas):
                if interrompido:
                    break
                if cap <= 0:
                    continue
                for tipo in (None, *tipos):
                    if interrompido:
                        break
                    if tipo == perfil[j][r]:
                        continue
                    proposta = [line.copy() for line in perfil]
                    proposta[j][r] = tipo
                    adicionar(proposta)
        quartos = [(j, r) for j in ordem_escolas
                   for r, cap in enumerate(instancia.escolas[j].capacidades_salas)
                   if cap > 0]
        pares = [(j, r, k, s) for p, (j, r) in enumerate(quartos)
                 for k, s in quartos[p + 1:]
                 if j != k and perfil[j][r] != perfil[k][s]]
        if prioridades_escolas is not None:
            pares.sort(key=lambda par: (-(prioridades_escolas[par[0]] +
                                          prioridades_escolas[par[2]]), par))
        else:
            pares = [par for par in pares
                     if capacidades[par[0]][par[1]] == capacidades[par[2]][par[3]]]
        if len(pares) > limite_swaps:
            pares = (pares[:limite_swaps] if prioridades_escolas is not None
                     else rng.sample(pares, limite_swaps))
        for j, r, k, s in pares:
            if interrompido:
                break
            proposta = [line.copy() for line in perfil]
            proposta[j][r], proposta[k][s] = proposta[k][s], proposta[j][r]
            adicionar(proposta)

    # None representa quarto vazio e não pode ser comparado diretamente com
    # os nomes string dos tipos quando duas propostas têm o mesmo limite.
    def chave_ordenacao(item):
        perfil_chave, limite = item
        perfil_texto = tuple(tuple("" if tipo is None else tipo for tipo in linha)
                             for linha in perfil_chave)
        return limite, perfil_texto

    return sorted(candidatos.items(), key=chave_ordenacao)


def busca_perfil_fluxo(instancia, solucao, tipos_por_local, seed=0, *,
                       tempo_limite, max_propostas=None,
                       temperatura_inicial=0.02, resfriamento=0.985,
                       estrategia='aleatoria', tempo_fluxo_proposta=25.0):
    """Busca perfis alternativos e otimiza cada perfil com fluxo capacitado.

    Perfis inviáveis ou cujo fluxo não conclui dentro do tempo da proposta são
    descartados. A saída é sempre a melhor solução validada encontrada.
    """
    if not isinstance(solucao, SolucaoSalas):
        raise ValueError('solucao deve ser SolucaoSalas')
    if isinstance(tipos_por_local, bool) or not isinstance(tipos_por_local, Integral) or tipos_por_local < 1:
        raise ValueError('tipos_por_local deve ser inteiro positivo')
    if tempo_limite is None or not math.isfinite(tempo_limite) or tempo_limite < 0:
        raise ValueError('tempo_limite deve ser finito e não negativo')
    if max_propostas is not None and (
            isinstance(max_propostas, bool) or not isinstance(max_propostas, Integral) or max_propostas < 0):
        raise ValueError('max_propostas deve ser None ou inteiro não negativo')
    if not math.isfinite(temperatura_inicial) or temperatura_inicial < 0:
        raise ValueError('temperatura_inicial deve ser finita e não negativa')
    if not math.isfinite(resfriamento) or not 0 < resfriamento <= 1:
        raise ValueError('resfriamento deve estar no intervalo (0, 1]')
    if estrategia not in ('aleatoria', 'menor_lb', 'custo_guiado'):
        raise ValueError("estrategia deve ser 'aleatoria', 'menor_lb' ou 'custo_guiado'")
    if not math.isfinite(tempo_fluxo_proposta) or tempo_fluxo_proposta <= 0:
        raise ValueError('tempo_fluxo_proposta deve ser finito e positivo')

    verificar_salas(instancia, solucao, tipos_por_local)
    inicio = time.perf_counter()
    deadline = inicio + tempo_limite
    rng = random.Random(seed)
    tipos = tuple(dict.fromkeys(c.tipo_prova for c in instancia.candidatos))
    candidatos_tipo = {t: np.asarray([i for i, c in enumerate(instancia.candidatos)
                                     if c.tipo_prova == t], dtype=int)
                       for t in tipos}
    matriz = np.asarray(instancia.matriz_distancias, dtype=float)
    perfil = _perfil_inicial(instancia, solucao, tipos_por_local, rng)
    atual, melhor = solucao, solucao
    custo_atual = custo_melhor = avaliar_salas(instancia, solucao)
    propostas = viaveis = rejeitados = aceitos = melhorias = fluxos = 0
    tipos_otimizados = aumentos = relaxacoes = 0
    motivo = 'estagnação_de_perfis'
    temperatura = temperatura_inicial * max(1.0, custo_atual)
    tentados = set()

    # Otimiza primeiro o perfil inicial, que pode incluir salas vazias já
    # reservadas ao tipo em K=1/2.
    perfil_inicial = otimizar_perfil_explicito(
        instancia, perfil, tipos_por_local,
        tempo_limite=max(0.0, min(20.0, deadline - time.perf_counter())))
    fluxos += 1
    tipos_otimizados += perfil_inicial.tipos_otimizados
    aumentos += perfil_inicial.aumentos_fluxo
    relaxacoes += perfil_inicial.relaxacoes
    if perfil_inicial.solucao is not None:
        atual = melhor = perfil_inicial.solucao
        custo_atual = custo_melhor = avaliar_salas(instancia, atual)
        viaveis += 1
        if custo_melhor < avaliar_salas(instancia, solucao) - 1e-10:
            melhorias += 1
    else:
        perfil = [linha.copy() for linha in perfil]

    while time.perf_counter() < deadline and (
            max_propostas is None or propostas < max_propostas):
        if estrategia in ('menor_lb', 'custo_guiado'):
            prazo_geracao = min(deadline, time.perf_counter() + 2.0)
            prioridades = (_prioridades_escolas(instancia, atual, matriz)
                           if estrategia == 'custo_guiado' else None)
            vizinhos = _vizinhos_lb(instancia, perfil, tipos_por_local, tipos,
                                    rng, candidatos_tipo, matriz, prazo_geracao,
                                    prioridades_escolas=prioridades)
            vizinhos = [item for item in vizinhos if item[0] not in tentados]
            if not vizinhos:
                tentados.clear()
                vizinhos = _vizinhos_lb(instancia, perfil, tipos_por_local, tipos,
                                        rng, candidatos_tipo, matriz,
                                        min(deadline, time.perf_counter() + 2.0),
                                        prioridades_escolas=prioridades)
            if not vizinhos:
                motivo = 'vizinhança_perfis_esgotada'
                break
            # RCL curta: usa o limite inferior para priorizar perfis promissores
            # e mantém uma pequena aleatorização entre os melhores candidatos.
            rcl = vizinhos[:min(5, len(vizinhos))]
            key, _ = rng.choice(rcl)
            tentados.add(key)
            proposta_perfil = [list(line) for line in key]
        else:
            proposta_perfil = _mutar(instancia, perfil, tipos_por_local, tipos, rng)
        propostas += 1
        if proposta_perfil is None:
            rejeitados += 1
            continue
        restante = max(0.0, deadline - time.perf_counter())
        if restante <= 0:
            motivo = 'limite_tempo'
            break
        # Uma tentativa longa pode terminar o fluxo de todo tipo; o limite
        # externo ainda encerra a vizinhança no teto global.
        tempo_fluxo = min(tempo_fluxo_proposta, restante)
        resultado = otimizar_perfil_explicito(
            instancia, proposta_perfil, tipos_por_local,
            tempo_limite=tempo_fluxo)
        fluxos += 1
        tipos_otimizados += resultado.tipos_otimizados
        aumentos += resultado.aumentos_fluxo
        relaxacoes += resultado.relaxacoes
        if resultado.solucao is None:
            rejeitados += 1
            if time.perf_counter() >= deadline:
                motivo = 'limite_tempo'
                break
            continue

        viaveis += 1
        custo_proposta = avaliar_salas(instancia, resultado.solucao)
        if custo_proposta < custo_melhor - 1e-10:
            melhor = resultado.solucao
            custo_melhor = custo_proposta
            melhorias += 1
        delta = custo_proposta - custo_atual
        if delta <= 1e-10 or (temperatura > 0 and rng.random() < math.exp(-delta / temperatura)):
            atual = resultado.solucao
            perfil = proposta_perfil
            custo_atual = custo_proposta
            aceitos += 1
        temperatura *= resfriamento

    else:
        if time.perf_counter() >= deadline:
            motivo = 'limite_tempo'
        elif max_propostas is not None and propostas >= max_propostas:
            motivo = 'limite_propostas'

    verificar_salas(instancia, melhor, tipos_por_local)
    return ResultadoBuscaPerfil(
        melhor, avaliar_salas(instancia, solucao), custo_melhor,
        propostas, viaveis, rejeitados, aceitos, melhorias, fluxos,
        tipos_otimizados, aumentos, relaxacoes, motivo)
