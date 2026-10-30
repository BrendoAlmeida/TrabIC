import unittest

from tests.test_construtivo_salas import instancia
from src.otimizacao.construtivo_add import construir_add
from src.otimizacao.salas import verificar_salas


class TestConstrutivoAdd(unittest.TestCase):
    def test_viabilidade_e_reproducibilidade(self):
        x = instancia(['M1', 'M1', 'M2', 'M2'], [[2, 2], [2, 2]],
                      [[1, 5], [2, 4], [4, 2], [5, 1]])
        for k in (1, 2, 3):
            a = construir_add(x, k, 42, reinicios=2)
            self.assertEqual(a, construir_add(x, k, 42, reinicios=2))
            self.assertEqual(a.status, 'viavel')
            verificar_salas(x, a.solucao, k)

    def test_fragmentacao_e_falha_nao_certifica_inviabilidade(self):
        x = instancia(['M1'] * 4 + ['M2'] * 2, [[3, 3]], [[1]] * 6)
        resultado = construir_add(x, 2, 0, reinicios=1, limite_configuracoes=10)
        self.assertEqual(resultado.status, 'falha_heuristica')
        self.assertIsNone(resultado.solucao)

    def test_ranking_lote_mais_proximo(self):
        x = instancia(['M1'] * 2, [[1], [1]], [[1, 7], [2, 5]])
        solucao = construir_add(x, 1, alpha=0, limite_reparacao=0).solucao
        self.assertEqual(solucao.alocacao, ((0, 0), (1, 0)))

    def test_argumentos_invalidos(self):
        x = instancia(['M1'], [[1]], [[1]])
        for kwargs in ({'tipos_por_local': 0}, {'tipos_por_local': True},
                       {'tipos_por_local': 1, 'reinicios': 0},
                       {'tipos_por_local': 1, 'alpha': -0.1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                construir_add(x, **kwargs)


if __name__ == '__main__':
    unittest.main()
