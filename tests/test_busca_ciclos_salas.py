import unittest

from tests.test_construtivo_salas import instancia
from src.otimizacao.busca_ciclos_salas import busca_ciclos_salas
from src.otimizacao.salas import SolucaoSalas, avaliar_salas, verificar_salas


class TestBuscaCiclosSalas(unittest.TestCase):
    def test_ciclo_troca_tipos_e_grupos_em_tres_escolas(self):
        x = instancia(['M1', 'M2', 'M3'], [[1], [1], [1]],
                      [[10, 0, 20], [20, 10, 0], [0, 20, 10]])
        s = SolucaoSalas(((0, 0), (1, 0), (2, 0)))
        r = busca_ciclos_salas(x, s, 1, max_ciclos=10)
        self.assertEqual(r.custo_inicial, 30)
        self.assertEqual(r.custo_final, 0)
        self.assertEqual(r.ciclos, 1)
        self.assertEqual(r.quartos_movidos, 3)
        verificar_salas(x, r.solucao, 1)

    def test_nao_degrada_com_tipos_compartilhados(self):
        x = instancia(['M1', 'M2', 'M1', 'M2'], [[2, 2], [2, 2]],
                      [[1, 5], [5, 1], [2, 4], [4, 2]])
        s = SolucaoSalas(((0, 0), (1, 0), (0, 1), (1, 1)))
        r = busca_ciclos_salas(x, s, 2, max_ciclos=10)
        verificar_salas(x, r.solucao, 2)
        self.assertLessEqual(r.custo_final, avaliar_salas(x, s))

    def test_limite_de_tempo_e_argumentos(self):
        x = instancia(['M1'], [[1]], [[1]])
        s = SolucaoSalas(((0, 0),))
        r = busca_ciclos_salas(x, s, 1, tempo_limite=0)
        self.assertEqual(r.solucao, s)
        self.assertEqual(r.motivo_parada, 'limite_tempo')
        with self.assertRaises(ValueError):
            busca_ciclos_salas(x, s, 1)
        with self.assertRaises(ValueError):
            busca_ciclos_salas(x, s, 1, max_ciclos=-1)


if __name__ == '__main__':
    unittest.main()
