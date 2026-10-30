import unittest

from tests.test_construtivo_salas import instancia
from src.otimizacao.cadeias import buscar_cadeias
from src.otimizacao.salas import SolucaoSalas, verificar_salas


class TestCadeias(unittest.TestCase):
    def test_ciclo_de_tres_salas_cheias(self):
        x = instancia(['M1'] * 3, [[1], [1], [1]],
                      [[10, 0, 20], [20, 10, 0], [0, 20, 10]])
        s = SolucaoSalas(((0, 0), (1, 0), (2, 0)))
        r = buscar_cadeias(x, s, 1, seed=4, max_comprimento=3, max_avaliacoes=1000)
        self.assertEqual(r.custo_final, 0)
        self.assertEqual(r.solucao.alocacao, ((1, 0), (2, 0), (0, 0)))
        verificar_salas(x, r.solucao, 1)
        self.assertEqual(r, buscar_cadeias(x, s, 1, seed=4, max_comprimento=3, max_avaliacoes=1000))
        curto = buscar_cadeias(x, s, 1, max_comprimento=2, max_avaliacoes=1000)
        self.assertEqual(curto.solucao, s)

    def test_limites_e_entrada_invalida(self):
        x = instancia(['M1', 'M2'], [[2], [1]], [[2, 1], [1, 2]])
        s = SolucaoSalas(((0, 0), (1, 0)))
        r = buscar_cadeias(x, s, 1, max_avaliacoes=0)
        self.assertEqual(r.solucao, s)
        self.assertEqual(r.motivo_parada, 'limite_avaliacoes')
        with self.assertRaises(ValueError):
            buscar_cadeias(x, SolucaoSalas(((0, 0), (0, 0))), 1)
        with self.assertRaises(ValueError):
            buscar_cadeias(x, s, 1, max_comprimento=0)

    def test_tipos_apos_cadeia_sao_verificados(self):
        x = instancia(['M1', 'M2', 'M1'], [[1, 1], [1]], [[5, 1], [5, 1], [1, 5]])
        s = SolucaoSalas(((0, 0), (0, 1), (1, 0)))
        for k in (2, 3):
            r = buscar_cadeias(x, s, k, max_avaliacoes=1000)
            verificar_salas(x, r.solucao, k)
            self.assertLessEqual(r.custo_final, r.custo_inicial)

if __name__ == '__main__':
    unittest.main()
