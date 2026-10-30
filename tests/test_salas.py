import unittest

from src.instancias.models import Candidato, Escola, Instancia
from src.otimizacao.salas import SolucaoSalas, avaliar_salas, resumo_salas, verificar_salas


def instancia(tipos, capacidades):
    candidatos = tuple(Candidato(f"0{i}", "", tipo, 0, 0) for i, tipo in enumerate(tipos))
    escolas = tuple(Escola(f"0{j}", "", 0, 0, tuple(salas))
                    for j, salas in enumerate(capacidades))
    matriz = tuple(tuple(float(i + j) for j in range(len(escolas))) for i in range(len(tipos)))
    return Instancia(candidatos, escolas, matriz)


class TestSalas(unittest.TestCase):
    def test_uma_ou_duas_provas_e_salas_vazias(self):
        x = instancia(["M1", "M1", "M2"], [[2, 1, 4], [2]])
        s = SolucaoSalas(((0, 0), (0, 0), (0, 1)))
        verificar_salas(x, s, 2)
        with self.assertRaises(ValueError):
            verificar_salas(x, s, 1)
        salas = resumo_salas(x, s, 2)
        self.assertEqual([s.tipo for s in salas], ["M1", "M2", None, None])
        self.assertEqual(salas[0].ids_candidatos, ("00", "01"))
        self.assertEqual(salas[0].id_escola, "00")
        self.assertEqual(avaliar_salas(x, s), 3.0)

    def test_homogeneidade_e_fragmentacao(self):
        x = instancia(["M1"] * 4 + ["M2"] * 2, [[3, 3]])
        for alocacao in (((0, 0),) * 4 + ((0, 1),) * 2,
                         ((0, 0),) * 3 + ((0, 1),) * 3):
            with self.assertRaises(ValueError):
                verificar_salas(x, SolucaoSalas(alocacao), 2)

    def test_indices_atribuicao_e_limite(self):
        x = instancia(["M1", "M2"], [[2, 2]])
        for alocacao in (((0, 0),), ((0, 0), (0, -1)), ((0, 0), (1, 0)),
                         ((0, 0), (0, True)), ((0, 0), 0)):
            with self.subTest(alocacao=alocacao), self.assertRaises(ValueError):
                verificar_salas(x, SolucaoSalas(alocacao), 2)
        s = SolucaoSalas(((0, 0), (0, 1)))
        for limite in (0, True, 1.5):
            with self.assertRaises(ValueError):
                verificar_salas(x, s, limite)
        verificar_salas(x, s, 3)

    def test_invariancia_das_posicoes_de_entrada(self):
        x = instancia(["M1", "M1"], [[1], [1]])
        s = SolucaoSalas(((1, 0), (0, 0)))
        verificar_salas(x, s, 1)
        self.assertEqual(avaliar_salas(x, s), 2.0)
        self.assertEqual(resumo_salas(x, s, 1)[0].indices_candidatos, (1,))


if __name__ == "__main__":
    unittest.main()
