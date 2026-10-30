import unittest

from tests.test_construtivo_salas import instancia
from src.otimizacao.busca_lns import busca_lns
from src.otimizacao.salas import SolucaoSalas, avaliar_salas, verificar_salas


class TestBuscaLNS(unittest.TestCase):
    def test_retorna_solucao_valida_e_nunca_piora_o_incumbente(self):
        x = instancia(['M1'] * 4, [[2], [2]],
                      [[1, 5], [5, 1], [2, 4], [4, 2]])
        s = SolucaoSalas(((1, 0), (0, 0), (1, 0), (0, 0)))
        r = busca_lns(x, s, 1, seed=19, max_iteracoes=100)
        self.assertLessEqual(r.custo_final, avaliar_salas(x, s))
        self.assertEqual(r, busca_lns(x, s, 1, seed=19, max_iteracoes=100))
        verificar_salas(x, r.solucao, 1)

    def test_limite_zero_preserva_solucao(self):
        x = instancia(['M1'], [[1]], [[1]])
        s = SolucaoSalas(((0, 0),))
        r = busca_lns(x, s, 1, seed=5, tempo_limite=0)
        self.assertEqual(r.solucao, s)
        self.assertEqual(r.motivo_parada, 'limite_tempo')
        verificar_salas(x, r.solucao, 1)

    def test_reparo_respeita_limite_de_tipos_e_sala_homogenea(self):
        x = instancia(['M1', 'M1', 'M2', 'M2'], [[1, 1], [1, 1]],
                      [[1, 3], [3, 1], [2, 4], [4, 2]])
        s = SolucaoSalas(((0, 0), (0, 1), (1, 0), (1, 1)))
        for k in (1, 2):
            with self.subTest(k=k):
                r = busca_lns(x, s, k, seed=3, max_iteracoes=30)
                verificar_salas(x, r.solucao, k)
                self.assertLessEqual(r.custo_final, r.custo_inicial)

    def test_validacao_dos_argumentos(self):
        x = instancia(['M1'], [[1]], [[1]])
        s = SolucaoSalas(((0, 0),))
        for kwargs in ({'max_iteracoes': -1}, {'tempo_limite': -1},
                       {'alpha': 1.1}, {'resfriamento': 0}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                busca_lns(x, s, 1, **kwargs)


if __name__ == '__main__':
    unittest.main()
