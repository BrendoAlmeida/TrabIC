import unittest

from tests.test_construtivo_salas import instancia
from src.otimizacao.busca_multiexchange import busca_multiexchange
from src.otimizacao.salas import SolucaoSalas, avaliar_salas, verificar_salas


class TestBuscaMultiExchange(unittest.TestCase):
    def test_ciclo_de_tres_escolas_cheias(self):
        x = instancia(['M1'] * 3, [[1], [1], [1]],
                      [[10, 0, 20], [20, 10, 0], [0, 20, 10]])
        s = SolucaoSalas(((0, 0), (1, 0), (2, 0)))
        r = busca_multiexchange(x, s, 1, max_ciclos=10)
        self.assertEqual(r.custo_inicial, 30)
        self.assertEqual(r.custo_final, 0)
        self.assertEqual(r.ciclos, 1)
        self.assertEqual(r.movimentos, 3)
        verificar_salas(x, r.solucao, 1)

    def test_ciclo_nao_cria_tipo_novo_em_escola(self):
        x = instancia(['M1', 'M1', 'M2', 'M2'], [[1, 1], [1, 1]],
                      [[8, 1], [1, 8], [8, 1], [1, 8]])
        s = SolucaoSalas(((0, 0), (1, 0), (0, 1), (1, 1)))
        r = busca_multiexchange(x, s, 2, max_ciclos=10)
        verificar_salas(x, r.solucao, 2)
        self.assertLessEqual(r.custo_final, r.custo_inicial)

    def test_limites_e_validacao(self):
        x = instancia(['M1'], [[1]], [[1]])
        s = SolucaoSalas(((0, 0),))
        r = busca_multiexchange(x, s, 1, tempo_limite=0)
        self.assertEqual(r.solucao, s)
        self.assertEqual(r.motivo_parada, 'limite_tempo')
        self.assertEqual(r.custo_final, avaliar_salas(x, s))
        with self.assertRaises(ValueError):
            busca_multiexchange(x, s, 1)
        with self.assertRaises(ValueError):
            busca_multiexchange(x, s, 1, max_ciclos=-1)


if __name__ == '__main__':
    unittest.main()
