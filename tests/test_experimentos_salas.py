import unittest
from fractions import Fraction

from experimentos import executar, limite_inferior_distancia
from tests.test_construtivo_salas import instancia


class TestExperimentosSalas(unittest.TestCase):
    def test_resultados_e_salas_vazias(self):
        x = instancia(['M1', 'M2'], [[1, 1, 1]], [[2], [3]])
        r = executar(x, tipos_por_local=2, seeds=[1, 2], reinicios=1,
                     limite_configuracoes=100, max_avaliacoes=30)
        self.assertEqual(r['resumo']['sucessos'], 2)
        self.assertEqual(r['resumo']['media'], 5)
        self.assertEqual(r['resumo']['desvio_padrao_populacional'], 0)
        self.assertEqual(r['parametros']['seeds'], [1, 2])
        for execucao in r['execucoes']:
            self.assertEqual(execucao['custo_final'], 5)
            self.assertEqual([s['tipo'] for s in execucao['salas']].count(None), 1)
            self.assertEqual(sum(len(s['ids_candidatos']) for s in execucao['salas']), 2)

    def test_metodo_add_e_reproducibilidade(self):
        x = instancia(['M1', 'M2'], [[1, 1], [1, 1]], [[1, 3], [3, 1]])
        a = executar(x, tipos_por_local=2, seeds=[42], metodo='add', reinicios=1,
                     max_avaliacoes=30)
        b = executar(x, tipos_por_local=2, seeds=[42], metodo='add', reinicios=1,
                     max_avaliacoes=30)
        self.assertEqual(a['parametros']['metodo'], 'add')
        self.assertEqual(a['execucoes'][0]['salas'], b['execucoes'][0]['salas'])
        self.assertEqual(a['execucoes'][0]['custo_final'], b['execucoes'][0]['custo_final'])

    def test_ablacoes_e_limite_inferior(self):
        x = instancia(['M1', 'M1'], [[1], [1]], [[1, 10], [1, 5]])
        for nome in ('construtivo', 'shift', 'shift_swap', 'vnd'):
            with self.subTest(ablacao=nome):
                r = executar(x, tipos_por_local=1, seeds=[42], reinicios=1,
                             max_avaliacoes=60, ablacao=nome)
                self.assertEqual(r['parametros']['ablacao'], nome)
                self.assertEqual(r['resumo']['limite_inferior'], 2)
                self.assertEqual(r['execucoes'][0]['gap_limite_inferior_pct'], 200)
                self.assertEqual(r['execucoes'][0]['busca'] is None, nome == 'construtivo')
        z = instancia(['M1'], [[1]], [[0]])
        r = executar(z, tipos_por_local=1, seeds=[1], ablacao='construtivo')
        self.assertIsNone(r['execucoes'][0]['gap_limite_inferior_pct'])

    def test_cadeias_opcional_registra_custo_e_parametros(self):
        x = instancia(['M1', 'M1'], [[1], [1]], [[1, 10], [1, 5]])
        r = executar(x, tipos_por_local=1, seeds=[42], reinicios=1,
                     max_avaliacoes=60, cadeias=True, comprimento_cadeia=2,
                     avaliacoes_cadeias=50)
        e = r['execucoes'][0]
        self.assertTrue(r['parametros']['cadeias'])
        self.assertEqual(r['parametros']['comprimento_cadeia'], 2)
        self.assertEqual(e['cadeias']['custo_final'], e['custo_final'])
        self.assertLessEqual(e['cadeias']['custo_final'], e['cadeias']['custo_inicial'])
        self.assertEqual(sum(len(s['ids_candidatos']) for s in e['salas']), 2)
        self.assertGreaterEqual(e['tempo_cadeias_s'], 0)

    def test_limite_inferior_nao_arredonda_para_cima(self):
        x = instancia(['M1', 'M1'], [[2]], [[0.1], [0.2]])
        exato = Fraction.from_float(0.1) + Fraction.from_float(0.2)
        self.assertLessEqual(Fraction.from_float(limite_inferior_distancia(x)), exato)

    def test_falha_sem_custo_valido(self):
        x = instancia(['M1'] * 4 + ['M2'] * 2, [[3, 3]], [[1]] * 6)
        r = executar(x, tipos_por_local=2, seeds=[0], reinicios=1, limite_configuracoes=0)
        self.assertEqual(r['resumo']['falhas'], 1)
        self.assertIsNone(r['resumo']['media'])
        self.assertIsNone(r['execucoes'][0]['salas'])
        self.assertEqual(r['execucoes'][0]['status'], 'falha_heuristica')

    def test_limite_global_zero_e_prazo_amplo(self):
        x = instancia(['M1', 'M2'], [[1, 1]], [[1], [2]])
        for metodo in ('regret', 'add'):
            with self.subTest(metodo=metodo):
                r = executar(x, tipos_por_local=2, seeds=[1], metodo=metodo,
                             tempo_total=0)
                self.assertEqual(r['execucoes'][0]['status'], 'tempo_limite')
                self.assertTrue(r['execucoes'][0]['limite_global_atingido'])
                self.assertIsNone(r['execucoes'][0]['custo_final'])
                r = executar(x, tipos_por_local=2, seeds=[1], metodo=metodo,
                             tempo_total=30, reinicios=1, max_avaliacoes=30)
                self.assertEqual(r['execucoes'][0]['status'], 'viavel')
                self.assertEqual(r['parametros']['tempo_total_s'], 30)
        for valor in (-1, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                executar(x, tipos_por_local=2, seeds=[1], tempo_total=valor)

    def test_semente_vazia_rejeitada(self):
        with self.assertRaises(ValueError):
            executar(instancia([], [], []), tipos_por_local=1, seeds=[])


if __name__ == '__main__':
    unittest.main()
