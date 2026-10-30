import unittest

from tests.test_construtivo_salas import instancia
from src.otimizacao.fluxo_perfil import otimizar_perfil
from src.otimizacao.salas import SolucaoSalas, avaliar_salas, verificar_salas


class TestFluxoPerfil(unittest.TestCase):
    def test_otimiza_atribuicao_capacitada_no_perfil_fixo(self):
        x = instancia(['M1'] * 4, [[2], [2]],
                      [[10, 0], [9, 0], [0, 10], [0, 9]])
        s = SolucaoSalas(((0, 0), (0, 0), (1, 0), (1, 0)))
        r = otimizar_perfil(x, s, 1, tempo_limite=2)
        self.assertEqual(r.status, 'perfil_otimizado')
        self.assertEqual(r.custo_final, 0)
        self.assertEqual(r.tipos_otimizados, 1)
        verificar_salas(x, r.solucao, 1)

    def test_preserva_tipos_e_nao_piora(self):
        x = instancia(['M1', 'M1', 'M2', 'M2'], [[2, 2], [2, 2]],
                      [[1, 5], [5, 1], [8, 2], [2, 8]])
        s = SolucaoSalas(((0, 0), (1, 0), (0, 1), (1, 1)))
        r = otimizar_perfil(x, s, 2, tempo_limite=2)
        verificar_salas(x, r.solucao, 2)
        self.assertLessEqual(r.custo_final, avaliar_salas(x, s))

    def test_zero_tempo_preserva_incumbente(self):
        x = instancia(['M1'], [[1]], [[1]])
        s = SolucaoSalas(((0, 0),))
        r = otimizar_perfil(x, s, 1, tempo_limite=0)
        self.assertEqual(r.solucao, s)
        self.assertEqual(r.status, 'limite_tempo')
        verificar_salas(x, r.solucao, 1)


if __name__ == '__main__':
    unittest.main()
