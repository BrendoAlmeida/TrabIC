import unittest

from tests.test_construtivo_salas import instancia
from src.otimizacao.salas import SolucaoSalas, avaliar_salas, verificar_salas
from src.otimizacao.busca_tabu import busca_tabu


class TestBuscaTabu(unittest.TestCase):
    def test_troca_em_salas_cheias_melhora(self):
        x = instancia(['M1', 'M1'], [[1], [1]], [[5, 1], [1, 5]])
        s = SolucaoSalas(((0, 0), (1, 0)))
        r = busca_tabu(x, s, 1, seed=4, tempo_limite=1, max_avaliacoes=1000)
        self.assertEqual(r.custo_final, 2)
        self.assertLess(r.custo_final, avaliar_salas(x, s))
        verificar_salas(x, r.solucao, 1)

    def test_reprodutibilidade_e_aspiracao_mantem_melhor(self):
        x = instancia(['M1'] * 4, [[2], [2]], [[1, 5], [5, 1], [2, 4], [4, 2]])
        s = SolucaoSalas(((0, 0), (1, 0), (0, 0), (1, 0)))
        a = busca_tabu(x, s, 1, seed=19, max_avaliacoes=1000)
        b = busca_tabu(x, s, 1, seed=19, max_avaliacoes=1000)
        self.assertEqual(a, b)
        self.assertLessEqual(a.custo_final, a.custo_inicial)
        verificar_salas(x, a.solucao, 1)

    def test_nao_ultrapassa_limite_de_tipos(self):
        x = instancia(['M1', 'M2'], [[1, 1], [1]], [[2, 1], [1, 2]])
        s = SolucaoSalas(((0, 0), (1, 0)))
        r = busca_tabu(x, s, 1, seed=5, tempo_limite=0.1, max_avaliacoes=1000)
        verificar_salas(x, r.solucao, 1)

    def test_para_apos_periodo_sem_melhorar_o_incumbente(self):
        x = instancia(['M1'], [[1], [1]], [[5, 0]])
        s = SolucaoSalas(((0, 0),))
        r = busca_tabu(x, s, 1, seed=5, tempo_limite=1,
                       tempo_sem_melhoria=0.02, max_avaliacoes=1_000_000,
                       tenure_min=1, tenure_max=1)
        self.assertEqual(r.custo_final, 0)
        self.assertEqual(r.motivo_parada, 'estagnacao')
        verificar_salas(x, r.solucao, 1)

    def test_limite_de_avaliacoes_pode_ser_desativado_com_teto_de_tempo(self):
        x = instancia(['M1'], [[1]], [[1]])
        s = SolucaoSalas(((0, 0),))
        r = busca_tabu(x, s, 1, tempo_limite=0, max_avaliacoes=None)
        self.assertEqual(r.motivo_parada, 'limite_tempo')
        with self.assertRaises(ValueError):
            busca_tabu(x, s, 1, max_avaliacoes=None)

    def test_validacao_dos_argumentos(self):
        x = instancia(['M1'], [[1]], [[1]])
        s = SolucaoSalas(((0, 0),))
        for kwargs in ({'max_avaliacoes': 0}, {'tenure_min': 4, 'tenure_max': 2},
                       {'tempo_limite': -1}, {'tempo_sem_melhoria': 0}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                busca_tabu(x, s, 1, **kwargs)


if __name__ == '__main__':
    unittest.main()
