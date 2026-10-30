import unittest

from src.instancias.models import Candidato, Escola, Instancia
from src.otimizacao.heuristica import otimizar
from src.otimizacao.modelo import avaliar, verificar
from src.otimizacao.salas import resumo_salas, verificar_salas


def instancia(custos, tipos, capacidades):
    cs = tuple(Candidato(str(i), "", t, 0, 0) for i, t in enumerate(tipos))
    es = tuple(Escola(str(j), "", 0, 0, (c,)) for j, c in enumerate(capacidades))
    return Instancia(cs, es, tuple(tuple(row) for row in custos), "km")


class TestHeuristica(unittest.TestCase):
    def test_verificador_indices_tipo_capacidade(self):
        x = instancia([[1, 2], [2, 1]], ["M1", "M2"], [1, 1])
        for al in ((0,), (0, 0), (-1, 1), (0, 2), (True, 1)):
            with self.assertRaises(ValueError): verificar(x, al, 1)
        with self.assertRaises(ValueError): verificar(x, (0, 1), 3)

    def test_capacidade_compartilhada_dois_tipos(self):
        x = instancia([[1], [1], [1]], ["M1", "M2", "M1"], [2])
        with self.assertRaises(ValueError): verificar(x, (0, 0, 0), 2)

    def test_determinismo_e_monotonicidade(self):
        x = instancia([[1, 4], [3, 1], [2, 5]], ["M1", "M2", "M1"], [2, 2])
        a = otimizar(x, 2, 42); b = otimizar(x, 2, 42)
        self.assertEqual(a, b)
        self.assertLessEqual(a.custo_final, a.custo_inicial + 1e-9)
        self.assertAlmostEqual(a.custo_final, avaliar(x, a.alocacao))
        verificar(x, a.alocacao, 2)
        verificar_salas(x, a.solucao, 2)

    def test_dois_tipos_em_salas_distintas_e_sala_vazia(self):
        cs = (Candidato("a", "", "M1", 0, 0), Candidato("b", "", "M2", 0, 0))
        escola = Escola("escola", "", 0, 0, (1, 1, 1))
        x = Instancia(cs, (escola,), ((1,), (2,)), "km")
        r = otimizar(x, 2, 42, reinicios=1)
        self.assertEqual(r.status, "viavel")
        salas = resumo_salas(x, r.solucao, 2)
        self.assertEqual({s.tipo for s in salas}, {"M1", "M2", None})
        self.assertEqual({id for s in salas for id in s.ids_candidatos}, {"a", "b"})
        self.assertEqual(r.alocacao, (0, 0))

    def test_typeswap_exemplo(self):
        # A solução final respeita o ótimo conhecido do exemplo.
        x = instancia([[1, 2], [1, 2], [1.1, 10], [1.1, 10]],
                      ["M1", "M1", "M2", "M2"], [2, 2])
        r = otimizar(x, 1, 0, reinicios=1)
        self.assertEqual(r.status, "viavel")
        self.assertLessEqual(r.custo_final, r.custo_inicial + 1e-9)
        verificar_salas(x, r.solucao, 1)
        self.assertAlmostEqual(r.custo_final, 6.2)

    def test_falha_nao_e_prova_de_inviabilidade(self):
        x = instancia([[1], [1]], ["M1", "M2"], [1])
        r = otimizar(x, 1, 2)
        self.assertEqual(r.status, "falha_heuristica")
        self.assertEqual(r.alocacao, ())


if __name__ == "__main__":
    unittest.main()
