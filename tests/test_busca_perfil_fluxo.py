import unittest
import random
import time

import numpy as np

from tests.test_construtivo_salas import instancia
from src.otimizacao.busca_perfil_fluxo import (
    busca_perfil_fluxo, _prioridades_escolas, _vizinhos_lb,
)
from src.otimizacao.salas import SolucaoSalas, verificar_salas


class TestBuscaPerfilFluxo(unittest.TestCase):
    def test_k1_swap_de_perfil_reduz_custo(self):
        x = instancia(['M1', 'M2'], [[1], [1]], [[10, 0], [0, 10]])
        s = SolucaoSalas(((0, 0), (1, 0)))
        r = busca_perfil_fluxo(x, s, 1, seed=3,
                               tempo_limite=1, max_propostas=20)
        self.assertEqual(r.custo_inicial, 20)
        self.assertEqual(r.custo_final, 0)
        self.assertGreater(r.perfis_viaveis, 0)
        verificar_salas(x, r.solucao, 1)

    def test_k1_ciclo_de_tres_escolas_muda_perfil(self):
        x = instancia(['M1', 'M2', 'M3'], [[1], [1], [1]],
                      [[10, 0, 20], [20, 10, 0], [0, 20, 10]])
        s = SolucaoSalas(((0, 0), (1, 0), (2, 0)))
        r = busca_perfil_fluxo(x, s, 1, seed=3,
                               tempo_limite=1, max_propostas=50)
        self.assertEqual(r.custo_final, 0)
        verificar_salas(x, r.solucao, 1)

    def test_vizinhanca_menor_limite_inferior_encontra_perfil(self):
        x = instancia(['M1', 'M2', 'M3'], [[1], [1], [1]],
                      [[10, 0, 20], [20, 10, 0], [0, 20, 10]])
        s = SolucaoSalas(((0, 0), (1, 0), (2, 0)))
        r = busca_perfil_fluxo(x, s, 1, seed=4, tempo_limite=1,
                               estrategia='menor_lb', max_propostas=10)
        self.assertEqual(r.custo_final, 0)
        verificar_salas(x, r.solucao, 1)

    def test_custo_guiado_prioriza_origem_e_destino_potenciais(self):
        x = instancia(['M1', 'M2'], [[1], [1], [1]],
                      [[10, 1, 50], [50, 50, 1]])
        s = SolucaoSalas(((0, 0), (2, 0)))
        prioridades = _prioridades_escolas(x, s)
        self.assertEqual(prioridades, [9.0, 9.0, 0.0])
        r = busca_perfil_fluxo(x, s, 1, seed=3, tempo_limite=1,
                               estrategia='custo_guiado', max_propostas=10)
        self.assertLessEqual(r.custo_final, r.custo_inicial)
        verificar_salas(x, r.solucao, 1)

    def test_vizinhanca_k2_ordena_empate_com_quarto_vazio(self):
        x = instancia(['M1', 'M2'], [[1, 1, 1]], [[1], [1]])
        perfil = [['M1', 'M1', 'M2']]
        vizinhos = _vizinhos_lb(
            x, perfil, 2, ('M1', 'M2'), random.Random(1),
            {'M1': np.asarray([0]), 'M2': np.asarray([1])},
            np.asarray(x.matriz_distancias, dtype=float),
            time.perf_counter() + 1.0)
        self.assertTrue(vizinhos)
        self.assertTrue(any(tipo is None for chave, _ in vizinhos for sala in chave
                            for tipo in sala))

    def test_k2_preserva_limite_e_viabilidade(self):
        x = instancia(['M1', 'M2', 'M1', 'M2'], [[2, 2], [2, 2]],
                      [[1, 5], [5, 1], [8, 2], [2, 8]])
        s = SolucaoSalas(((0, 0), (1, 0), (0, 1), (1, 1)))
        r = busca_perfil_fluxo(x, s, 2, seed=7,
                               tempo_limite=1, estrategia='custo_guiado',
                               max_propostas=10)
        self.assertLessEqual(r.custo_final, r.custo_inicial)
        verificar_salas(x, r.solucao, 2)

    def test_limite_zero_preserva_incumbente(self):
        x = instancia(['M1'], [[1]], [[1]])
        s = SolucaoSalas(((0, 0),))
        r = busca_perfil_fluxo(x, s, 1, tempo_limite=0)
        self.assertEqual(r.solucao, s)
        self.assertEqual(r.motivo_parada, 'limite_tempo')


if __name__ == '__main__':
    unittest.main()
