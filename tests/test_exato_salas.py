import itertools
from fractions import Fraction
import unittest
from unittest.mock import patch

from src.instancias.models import Candidato, Escola, Instancia
from src.otimizacao.exato_salas import resolver_exato_salas
from src.otimizacao.salas import verificar_salas


def caso(tipos, salas, matriz):
    cs = tuple(Candidato(str(i), 'cep', t, 0, 0) for i, t in enumerate(tipos))
    es = tuple(Escola(str(j), 'cep', 0, 0, tuple(c)) for j, c in enumerate(salas))
    return Instancia(cs, es, tuple(tuple(row) for row in matriz), 'km')


class TestExatoSalas(unittest.TestCase):
    def test_compara_enumeracao_independente(self):
        x = caso(['M1', 'M2', 'M1', 'M2'], [[2, 1], [2, 1]],
                 [[1, 3], [2, 1], [3, 1], [2, 4]])
        possiveis = [(j, r) for j, e in enumerate(x.escolas) for r in range(len(e.capacidades_salas))]
        for k in (1, 2, 3):
            valores = []
            for aloc in itertools.product(possiveis, repeat=len(x.candidatos)):
                ocup = {(j, r): [] for j, r in possiveis}
                for i, pos in enumerate(aloc):
                    ocup[pos].append(i)
                if any(len(m) > x.escolas[j].capacidades_salas[r] or
                       len({x.candidatos[i].tipo_prova for i in m}) > 1
                       for (j, r), m in ocup.items()):
                    continue
                if any(len({x.candidatos[i].tipo_prova for i, p in enumerate(aloc) if p[0] == j}) > k
                       for j in range(len(x.escolas))):
                    continue
                valores.append(sum(x.matriz_distancias[i][p[0]] for i, p in enumerate(aloc)))
            resultado = resolver_exato_salas(x, k)
            self.assertEqual(resultado.status, 'otimo')
            self.assertEqual(resultado.custo, min(valores))
            verificar_salas(x, resultado.solucao, k)
            self.assertEqual(resultado.bound, resultado.custo)
            self.assertEqual(resultado.gap, 0)

    def test_bound_certificado_nao_ultrapassa_otimo_exato(self):
        x = caso(['M1', 'M1'], [[2]], [[0.1], [0.2]])
        resultado = resolver_exato_salas(x, 1)
        exato = Fraction.from_float(0.1) + Fraction.from_float(0.2)
        self.assertEqual(resultado.status, 'otimo')
        self.assertGreater(Fraction.from_float(resultado.custo), exato)
        self.assertLessEqual(Fraction.from_float(resultado.bound), exato)
        self.assertLessEqual(Fraction.from_float(resultado.lower_bound), exato)

    def test_fragmentacao_prova_inviabilidade(self):
        x = caso(['M1'] * 4 + ['M2'] * 2, [[3, 3]], [[1]] * 6)
        resultado = resolver_exato_salas(x, 2)
        self.assertEqual(resultado.status, 'inviavel')
        self.assertIsNone(resultado.solucao)

    def test_limites_e_interrupcao_nao_certificam(self):
        x = caso(['M1', 'M1', 'M2'], [[2, 2], [2]], [[1, 2]] * 3)
        for resultado in (resolver_exato_salas(x, 2, max_candidatos=1),
                          resolver_exato_salas(x, 2, max_combinacoes=1),
                          resolver_exato_salas(x, 2, tempo_limite=1e-12)):
            self.assertEqual(resultado.status, 'limite')
            self.assertIsNone(resultado.bound)
            self.assertIsNone(resultado.gap)
        ticks = iter(i * 0.05 for i in range(1000))
        with patch('src.otimizacao.exato_salas.monotonic', side_effect=lambda: next(ticks)):
            resultado = resolver_exato_salas(x, 2, tempo_limite=0.32)
        self.assertEqual(resultado.status, 'limite')
        self.assertIsNotNone(resultado.solucao)
        self.assertIsNone(resultado.bound)
        verificar_salas(x, resultado.solucao, 2)


if __name__ == '__main__':
    unittest.main()
