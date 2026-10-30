import unittest

from tests.test_construtivo_salas import instancia
from src.otimizacao.salas import SolucaoSalas, avaliar_salas, verificar_salas
from src.otimizacao.busca_local import delta_shift, delta_swap, vnd


class TestBuscaLocalSalas(unittest.TestCase):
    def test_deltas_contra_avaliacao_integral(self):
        x = instancia(['M1', 'M1', 'M1'], [[2], [2]], [[4, 1], [1, 7], [3, 2]])
        s = SolucaoSalas(((0, 0), (1, 0), (0, 0)))
        shift = SolucaoSalas(((1, 0), (1, 0), (0, 0)))
        swap = SolucaoSalas(((1, 0), (0, 0), (0, 0)))
        self.assertAlmostEqual(delta_shift(x, 0, 0, 1), avaliar_salas(x, shift) - avaliar_salas(x, s))
        self.assertAlmostEqual(delta_swap(x, 0, 1, 0, 1), avaliar_salas(x, swap) - avaliar_salas(x, s))

    def test_shift_usa_sala_vazia(self):
        x = instancia(['M1'], [[1], [1]], [[5, 1]])
        r = vnd(x, SolucaoSalas(((0, 0),)), 1)
        self.assertEqual(r.custo_final, 1)
        self.assertEqual(r.solucao.alocacao, ((1, 0),))
        verificar_salas(x, r.solucao, 1)

    def test_swap_em_salas_cheias(self):
        x = instancia(['M1', 'M1'], [[1], [1]], [[5, 1], [1, 5]])
        r = vnd(x, SolucaoSalas(((0, 0), (1, 0))), 1)
        self.assertEqual(r.custo_final, 2)
        verificar_salas(x, r.solucao, 1)

    def test_reconfiguracao_troca_tipos_em_salas_cheias(self):
        x = instancia(['M1', 'M1', 'M2', 'M2'], [[2], [2]],
                      [[1, 2], [1, 2], [1.1, 10], [1.1, 10]])
        s = SolucaoSalas(((0, 0), (0, 0), (1, 0), (1, 0)))
        r = vnd(x, s, 1)
        self.assertAlmostEqual(r.custo_final, 6.2)
        self.assertGreater(r.melhorias, 0)
        verificar_salas(x, r.solucao, 1)

    def test_reconstrucao_redistribui_varios_candidatos(self):
        x = instancia(['M1'] * 5, [[3], [2]],
                      [[9, 1], [9, 1], [1, 9], [1, 9], [1, 9]])
        s = SolucaoSalas(((0, 0), (0, 0), (0, 0), (1, 0), (1, 0)))
        # Uma avaliação de swap não consegue realizar as duas trocas necessárias.
        r = vnd(x, s, 1, seed=1, max_avaliacoes=3)
        self.assertEqual(r.melhorias, 1)
        self.assertEqual(r.custo_final, 5)
        verificar_salas(x, r.solucao, 1)

    def test_cotas_zero_nao_afirmam_vizinhancas_examinadas(self):
        x = instancia(['M1'], [[1]], [[1]])
        s = SolucaoSalas(((0, 0),))
        for limite in (1, 2):
            r = vnd(x, s, 1, max_avaliacoes=limite)
            self.assertIn(r.motivo_parada, ('limite_avaliacoes', 'limite_vizinhanca'))

    def test_selecao_de_vizinhancas_para_ablacao(self):
        x = instancia(['M1', 'M1'], [[1], [1]], [[5, 1], [1, 5]])
        s = SolucaoSalas(((0, 0), (1, 0)))
        apenas_shift = vnd(x, s, 1, vizinhancas=('shift',))
        com_swap = vnd(x, s, 1, vizinhancas=('shift', 'swap'))
        self.assertEqual(apenas_shift.custo_final, 10)
        self.assertEqual(com_swap.custo_final, 2)
        self.assertEqual(vnd(x, s, 1), vnd(x, s, 1, vizinhancas=('shift', 'swap', 'reconfiguracao')))
        for selecionadas in ((), ('shift', 'shift'), ('inexistente',), 'shift'):
            with self.subTest(vizinhanças=selecionadas), self.assertRaises(ValueError):
                vnd(x, s, 1, vizinhancas=selecionadas)

    def test_limite_e_reprodutibilidade(self):
        x = instancia(['M1', 'M1'], [[1], [1]], [[5, 1], [1, 5]])
        s = SolucaoSalas(((0, 0), (1, 0)))
        r = vnd(x, s, 2, seed=2, max_avaliacoes=0)
        self.assertEqual(r.solucao, s)
        self.assertEqual(r.motivo_parada, 'limite_avaliacoes')
        self.assertEqual(vnd(x, s, 2, seed=4), vnd(x, s, 2, seed=4))
        with self.assertRaises(ValueError):
            vnd(x, SolucaoSalas(((0, 0), (0, 0))), 2)


if __name__ == '__main__':
    unittest.main()
