import unittest
from unittest.mock import patch

from src.instancias.models import Candidato, Escola, Instancia
from src.otimizacao.construtivo import construir
from src.otimizacao.salas import verificar_salas


def instancia(tipos, salas, distancias):
    candidatos = tuple(Candidato(str(i), '', t, 0, 0) for i, t in enumerate(tipos))
    escolas = tuple(Escola(str(j), '', 0, 0, tuple(caps)) for j, caps in enumerate(salas))
    return Instancia(candidatos, escolas, tuple(tuple(d) for d in distancias))


class TestConstrutivoSalas(unittest.TestCase):
    def test_viavel_reprodutivel_em_ambos_os_limites(self):
        x = instancia(['M1', 'M1', 'M2', 'M2'], [[2, 2], [2, 2]],
                      [[1, 5], [2, 4], [4, 2], [5, 1]])
        for k in (1, 2, 3):
            a = construir(x, k, 42)
            self.assertEqual(a, construir(x, k, 42))
            self.assertEqual(a.status, 'viavel')
            verificar_salas(x, a.solucao, k)

    def test_fragmentacao_e_falha_nao_certifica_inviabilidade(self):
        x = instancia(['M1'] * 4 + ['M2'] * 2, [[3, 3]], [[1]] * 6)
        r = construir(x, 2, 0, reinicios=2)
        self.assertEqual(r.status, 'falha_heuristica')
        self.assertIsNone(r.solucao)
        self.assertEqual(r.tentativas, 2)

    def test_candidato_com_unica_escola_e_sala_vazia(self):
        x = instancia(['M1', 'M2'], [[1], [1]], [[0, 5], [0, 1]])
        s = construir(x, 1, 3).solucao
        verificar_salas(x, s, 1)
        self.assertEqual(len(s.alocacao), 2)

    def test_recuperacao_reconfigura_salas(self):
        x = instancia(['M1', 'M1', 'M2', 'M1', 'M3e', 'M1', 'M2'],
                      [[2, 2], [1, 1], [2, 2]],
                      [[4, 2, 6], [4, 6, 5], [6, 3, 2], [1, 2, 2],
                       [3, 3, 0], [7, 2, 4], [4, 0, 2]])
        sem = construir(x, 1, 0, reinicios=1, limite_configuracoes=0)
        com = construir(x, 1, 0, reinicios=1, limite_configuracoes=3000)
        self.assertEqual(sem.status, 'falha_heuristica')
        self.assertEqual(com.status, 'viavel')
        self.assertGreater(com.reparacoes, 0)
        verificar_salas(x, com.solucao, 1)

    def test_recupera_antes_do_proximo_reinicio(self):
        x = instancia(['M1', 'M1', 'M2', 'M1', 'M3e', 'M1', 'M2'],
                      [[2, 2], [1, 1], [2, 2]],
                      [[4, 2, 6], [4, 6, 5], [6, 3, 2], [1, 2, 2],
                       [3, 3, 0], [7, 2, 4], [4, 0, 2]])
        from src.otimizacao.construtivo import _recuperar_configuracao
        with patch('src.otimizacao.construtivo._recuperar_configuracao',
                   wraps=_recuperar_configuracao) as recuperar:
            resultado = construir(x, 1, 0, reinicios=3, limite_configuracoes=3000)
        self.assertEqual(resultado.status, 'viavel')
        self.assertEqual(resultado.tentativas, 1)
        self.assertEqual(recuperar.call_count, 1)
        verificar_salas(x, resultado.solucao, 1)

    def test_opcoes_invalidas(self):
        x = instancia(['M1'], [[1]], [[1]])
        for kwargs in ({'tipos_por_local': 0}, {'tipos_por_local': True},
                       {'tipos_por_local': 1, 'reinicios': 0},
                       {'tipos_por_local': 1, 'alpha': -0.1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                construir(x, **kwargs)


if __name__ == '__main__':
    unittest.main()
