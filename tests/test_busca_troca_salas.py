import unittest

from tests.test_construtivo_salas import instancia
from src.otimizacao.busca_troca_salas import busca_troca_salas
from src.otimizacao.salas import SolucaoSalas, avaliar_salas, verificar_salas


class TestBuscaTrocaSalas(unittest.TestCase):
    def test_troca_de_tipos_entre_escolas_cheias(self):
        x = instancia(['M1', 'M2'], [[1], [1]], [[10, 0], [0, 10]])
        s = SolucaoSalas(((0, 0), (1, 0)))
        r = busca_troca_salas(x, s, 1, max_trocas=10)
        self.assertEqual(r.custo_inicial, 20)
        self.assertEqual(r.custo_final, 0)
        self.assertEqual(r.trocas, 1)
        verificar_salas(x, r.solucao, 1)

    def test_nao_degrada_e_preserva_k_dois(self):
        x = instancia(['M1', 'M2', 'M1', 'M2'], [[2], [2]],
                      [[1, 5], [5, 1], [2, 4], [4, 2]])
        s = SolucaoSalas(((0, 0), (1, 0), (0, 0), (1, 0)))
        r = busca_troca_salas(x, s, 2, max_trocas=10)
        self.assertLessEqual(r.custo_final, avaliar_salas(x, s))
        verificar_salas(x, r.solucao, 2)

    def test_limites(self):
        x = instancia(['M1'], [[1]], [[1]])
        s = SolucaoSalas(((0, 0),))
        r = busca_troca_salas(x, s, 1, tempo_limite=0)
        self.assertEqual(r.solucao, s)
        self.assertEqual(r.motivo_parada, 'limite_tempo')
        with self.assertRaises(ValueError):
            busca_troca_salas(x, s, 1)


if __name__ == '__main__':
    unittest.main()
