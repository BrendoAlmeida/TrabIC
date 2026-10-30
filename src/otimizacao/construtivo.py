"""Construtivo regret por sala, com reparação limitada e reinícios.

O fracasso não certifica inviabilidade. Cada solução retornada é verificada
independentemente; índices seguem a ordem dos arquivos.
"""
from dataclasses import dataclass
from numbers import Integral
import random
import math
import time

import numpy as np

from .salas import SolucaoSalas, verificar_salas


@dataclass(frozen=True)
class ResultadoConstrucao:
    status: str
    solucao: SolucaoSalas | None
    tentativas: int
    reparacoes: int
    motivo: str


def construir(instancia, tipos_por_local: int, seed: int = 0, *, alpha: float = 0.2,
              reinicios: int = 10, limite_reparacao: int = 100,
              limite_configuracoes: int = 30000, prazo: float | None = None) -> ResultadoConstrucao:
    """Constrói uma solução. `reinicios` é o número total máximo de tentativas.

    A lista restrita contém regrets >= máximo - alpha*(máximo-mínimo).
    Uma escola é alternativa se tiver uma sala disponível para o tipo. Na
    reparação, desloca-se um candidato e tenta-se inserir o candidato bloqueado;
    tentativas malsucedidas não alteram o estado. A ordem aleatória vem da seed.
    """
    if isinstance(tipos_por_local, bool) or not isinstance(tipos_por_local, Integral) or tipos_por_local < 1:
        raise ValueError('tipos_por_local deve ser um inteiro positivo')
    if not 0 <= alpha <= 1:
        raise ValueError('alpha deve estar entre 0 e 1')
    for nome, valor, minimo in [('reinicios', reinicios, 1), ('limite_reparacao', limite_reparacao, 0), ('limite_configuracoes', limite_configuracoes, 0)]:
        if isinstance(valor, bool) or not isinstance(valor, Integral) or valor < minimo:
            raise ValueError(f'{nome} deve ser um inteiro >= {minimo}')
    if prazo is not None and not math.isfinite(prazo):
        raise ValueError('prazo deve ser finito')
    expirou = lambda: prazo is not None and time.perf_counter() >= prazo
    if expirou():
        return ResultadoConstrucao('tempo_limite', None, 0, 0, 'prazo global esgotado')
    n, m = len(instancia.candidatos), len(instancia.escolas)
    if n == 0:
        solucao = SolucaoSalas(())
        verificar_salas(instancia, solucao, tipos_por_local)
        return ResultadoConstrucao('viavel', solucao, 1, 0, '')
    matriz = np.asarray(instancia.matriz_distancias, dtype=np.float64)
    if matriz.shape != (n, m) or not np.isfinite(matriz).all():
        raise ValueError('Matriz de distâncias inválida')
    if m == 0:
        return ResultadoConstrucao('falha_heuristica', None, 0, 0, 'nenhuma escola disponível')
    tipos = [c.tipo_prova for c in instancia.candidatos]
    tipos_distintos = tuple(dict.fromkeys(tipos))
    codigos = np.array([tipos_distintos.index(t) for t in tipos], dtype=int)
    capacidades = [e.capacidades_salas for e in instancia.escolas]
    rng = random.Random(seed)
    reparacoes_total = 0
    for tentativa in range(1, reinicios + 1):
        if expirou():
            return ResultadoConstrucao('tempo_limite', None, tentativa - 1, reparacoes_total, 'prazo global esgotado')
        al = [None] * n
        membros = [[set() for _ in caps] for caps in capacidades]
        tipos_salas = [[None] * len(caps) for caps in capacidades]
        contagens = [dict() for _ in capacidades]
        nao_alocados = np.ones(n, dtype=bool)

        def opcoes(tipo, excluida=None):
            resultado = []
            for j, caps in enumerate(capacidades):
                if tipo not in contagens[j] and len(contagens[j]) >= tipos_por_local:
                    continue
                salas = [r for r, cap in enumerate(caps)
                         if (j, r) != excluida and len(membros[j][r]) < cap
                         and tipos_salas[j][r] in (None, tipo)]
                if salas:
                    # Sala com menor folga evita fragmentar capacidade.
                    r = min(salas, key=lambda k: (caps[k] - len(membros[j][k]), k))
                    resultado.append((j, r))
            return resultado

        def inserir(i, pos):
            j, r = pos
            tipo = tipos[i]
            membros[j][r].add(i)
            tipos_salas[j][r] = tipo
            contagens[j][tipo] = contagens[j].get(tipo, 0) + 1
            al[i] = pos

        def remover(i):
            j, r = al[i]
            tipo = tipos[i]
            membros[j][r].remove(i)
            contagens[j][tipo] -= 1
            if contagens[j][tipo] == 0:
                del contagens[j][tipo]
            if not membros[j][r]:
                tipos_salas[j][r] = None
            al[i] = None
            return j, r

        while nao_alocados.any():
            if expirou():
                return ResultadoConstrucao('tempo_limite', None, tentativa, reparacoes_total, 'prazo global esgotado')
            disponiveis = [opcoes(t) for t in tipos_distintos]
            pendentes = np.flatnonzero(nao_alocados)
            tipos_bloqueados = np.array([not op for op in disponiveis], dtype=bool)
            bloqueados = pendentes[tipos_bloqueados[codigos[pendentes]]]
            if bloqueados.size:
                i = int(bloqueados[0])
                deslocados = [k for k in range(n) if al[k] is not None]
                rng.shuffle(deslocados)
                corrigiu = False
                for k in deslocados[:limite_reparacao]:
                    if expirou():
                        return ResultadoConstrucao('tempo_limite', None, tentativa, reparacoes_total, 'prazo global esgotado')
                    origem = remover(k)
                    alternativas = opcoes(tipos[k], excluida=origem)
                    alternativas.sort(key=lambda pos: (matriz[k, pos[0]], pos))
                    for destino in alternativas:
                        inserir(k, destino)
                        lugares = opcoes(tipos[i])
                        if lugares:
                            pos = min(lugares, key=lambda p: (matriz[i, p[0]], p))
                            inserir(i, pos)
                            nao_alocados[i] = False
                            reparacoes_total += 1
                            corrigiu = True
                            break
                        remover(k)
                    if corrigiu:
                        break
                    inserir(k, origem)
                if not corrigiu:
                    break
                continue

            # Apenas mudanças no estado das salas alteram a disponibilidade.
            # Ordenar escolas de cada candidato permite achar as duas primeiras.
            regrets = np.empty(len(pendentes), dtype=float)
            melhores = np.empty(len(pendentes), dtype=int)
            unicas = np.zeros(len(pendentes), dtype=bool)
            for codigo, op in enumerate(disponiveis):
                indices = np.flatnonzero(codigos[pendentes] == codigo)
                if not len(indices):
                    continue
                escolas = np.array([j for j, _ in op], dtype=int)
                custos = matriz[np.ix_(pendentes[indices], escolas)]
                minimo_idx = np.argmin(custos, axis=1)
                melhores[indices] = escolas[minimo_idx]
                if len(escolas) == 1:
                    unicas[indices] = True
                    regrets[indices] = 0
                else:
                    dois = np.partition(custos, 1, axis=1)[:, :2]
                    regrets[indices] = dois[:, 1] - dois[:, 0]
            if unicas.any():
                candidatos_rcl = np.flatnonzero(unicas)
            else:
                limite = regrets.max() - alpha * (regrets.max() - regrets.min())
                candidatos_rcl = np.flatnonzero(regrets >= limite)
            escolhido = int(candidatos_rcl[rng.randrange(len(candidatos_rcl))])
            i = int(pendentes[escolhido])
            j = int(melhores[escolhido])
            pos = next(pos for pos in disponiveis[codigos[i]] if pos[0] == j)
            inserir(i, pos)
            nao_alocados[i] = False
        if not nao_alocados.any():
            solucao = SolucaoSalas(tuple(al))
            verificar_salas(instancia, solucao, tipos_por_local)
            return ResultadoConstrucao('viavel', solucao, tentativa, reparacoes_total, '')
        # Uma nova construção inteira pode consumir o prazo antes de atingir
        # a recuperação estrutural. Tente-a imediatamente após cada bloqueio.
        reserva = _recuperar_configuracao(instancia, tipos_por_local, rng, limite_configuracoes, prazo=prazo)
        if reserva is not None:
            verificar_salas(instancia, reserva, tipos_por_local)
            return ResultadoConstrucao('viavel', reserva, tentativa, reparacoes_total + 1, '')
        if expirou():
            return ResultadoConstrucao('tempo_limite', None, tentativa, reparacoes_total, 'prazo global esgotado')
    return ResultadoConstrucao('falha_heuristica', None, reinicios, reparacoes_total,
                              'tentativas esgotadas com candidatos bloqueados')


def _recuperar_configuracao(instancia, limite_tipos, rng, limite, *, prazo=None):
    """Busca limitada por capacidades por tipo e depois aloca candidatos.

    Reconfigura salas/escolas com annealing sobre déficit de capacidade; não
    prova inviabilidade quando o limite é esgotado.
    """
    candidatos = instancia.candidatos
    tipos = tuple(dict.fromkeys(c.tipo_prova for c in candidatos))
    if not tipos or not limite:
        return None
    caps = [tuple(e.capacidades_salas) for e in instancia.escolas]
    escolas_uteis = [j for j, c in enumerate(caps) if c]
    if not escolas_uteis:
        return None
    demanda = [sum(c.tipo_prova == t for c in candidatos) for t in tipos]
    codigos = [tipos.index(c.tipo_prova) for c in candidatos]
    config = []
    totais = [0] * len(tipos)
    for salas in caps:
        tipo = rng.randrange(len(tipos))
        config.append([tipo] * len(salas))
        totais[tipo] += sum(salas)
    def deficit(totais):
        return sum(max(0, d - q) for d, q in zip(demanda, totais))
    atual = deficit(totais)
    for passo in range(limite + 1):
        if prazo is not None and time.perf_counter() >= prazo:
            return None
        if atual == 0:
            break
        if passo == limite:
            return None
        j = rng.choice(escolas_uteis)
        antigo = config[j][:]
        tipo = rng.randrange(len(tipos))
        novo = antigo[:]
        if limite_tipos == 1 or rng.random() < 0.25:
            novo = [tipo] * len(novo)
        else:
            novo[rng.randrange(len(novo))] = tipo
        if len(set(novo)) > limite_tipos or novo == antigo:
            continue
        for r, cap in enumerate(caps[j]):
            totais[antigo[r]] -= cap
            totais[novo[r]] += cap
        novo_deficit = deficit(totais)
        temperatura = max(0.1, 0.03 * len(candidatos) * (1 - passo / max(1, limite)))
        if novo_deficit <= atual or rng.random() < math.exp((atual - novo_deficit) / temperatura):
            config[j] = novo
            atual = novo_deficit
        else:
            for r, cap in enumerate(caps[j]):
                totais[novo[r]] -= cap
                totais[antigo[r]] += cap
    # Com capacidade reservada suficiente para cada tipo, uma alocação gulosa
    # por custo é sempre completa: todos os candidatos de um tipo têm acesso
    # às mesmas salas desse tipo, com custos fornecidos por escola.
    ocup = [[0] * len(c) for c in caps]
    al = [None] * len(candidatos)
    matriz = np.asarray(instancia.matriz_distancias, dtype=float)
    for t in range(len(tipos)):
        indices = [i for i, codigo in enumerate(codigos) if codigo == t]
        escolas = [j for j, salas in enumerate(config) if t in salas]
        if len(escolas) > 1:
            subset = matriz[np.ix_(indices, escolas)]
            pares = np.partition(subset, 1, axis=1)[:, :2]
            indices = [indices[k] for k in np.argsort(-(pares[:, 1] - pares[:, 0]), kind='stable')]
        for i in indices:
            if prazo is not None and time.perf_counter() >= prazo:
                return None
            possibilidades = [(matriz[i, j], j, r) for j in escolas for r, tipo in enumerate(config[j])
                              if tipo == t and ocup[j][r] < caps[j][r]]
            if not possibilidades:
                return None
            _, j, r = min(possibilidades)
            ocup[j][r] += 1
            al[i] = (j, r)
    return SolucaoSalas(tuple(al))
