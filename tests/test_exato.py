import itertools
import math
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch
from contextlib import redirect_stdout
from io import StringIO

from src.instancias.io import ler_instancia
from src.instancias.models import Candidato, Escola, Instancia
from src.otimizacao.exato import resolver_exato
from brute_force import main as cli_main


def instancia_pequena(matriz=None, capacidades=(2, 2, 2)):
    candidatos = tuple(
        Candidato(f"C{i}", f"000{i}", tipo, 0, 0)
        for i, tipo in enumerate(("M1", "M2", "M1", "M2"), 1)
    )
    escolas = tuple(
        Escola(f"E{i}", f"111{i}", 0, 0, (capacidade,))
        for i, capacidade in enumerate(capacidades, 1)
    )
    if matriz is None:
        matriz = ((1, 3, 2), (2, 1, 4), (3, 2, 1), (4, 1, 2))
    return Instancia(candidatos, escolas, matriz, "km")


def custo_bruto(instancia, aloc):
    return sum(instancia.matriz_distancias[i][j] for i, j in enumerate(aloc))


def viavel_bruto(instancia, aloc, limite_tipos):
    for j, escola in enumerate(instancia.escolas):
        membros = [i for i, destino in enumerate(aloc) if destino == j]
        if len(membros) > escola.capacidade:
            return False
        if len({instancia.candidatos[i].tipo_prova for i in membros}) > limite_tipos:
            return False
    return True


class TesteExato(unittest.TestCase):
    def test_compara_enumeracao_independente_nos_dois_modos(self):
        instancia = instancia_pequena()
        for limite in (1, 2):
            esperados = [
                custo_bruto(instancia, a)
                for a in itertools.product(range(len(instancia.escolas)), repeat=len(instancia.candidatos))
                if viavel_bruto(instancia, a, limite)
            ]
            resultado = resolver_exato(instancia, limite)
            self.assertEqual(resultado.status, "otimo")
            self.assertEqual(resultado.custo, min(esperados))
            self.assertTrue(viavel_bruto(instancia, resultado.alocacao, limite))

    def test_inviavel(self):
        resultado = resolver_exato(instancia_pequena(capacidades=(1, 1, 1)), 1)
        self.assertEqual(resultado.status, "inviavel")
        self.assertIsNone(resultado.alocacao)

    def test_limites_nunca_reportam_otimo(self):
        instancia = instancia_pequena()
        por_tamanho = resolver_exato(instancia, 1, max_candidatos=2)
        por_estimativa = resolver_exato(instancia, 1, max_combinacoes=10)
        por_tempo = resolver_exato(instancia, 1, tempo_limite=1e-12,
                                   max_candidatos=None, max_combinacoes=None)
        for resultado in (por_tamanho, por_estimativa, por_tempo):
            self.assertEqual(resultado.status, "limite")
            self.assertIsNone(resultado.gap)

    def test_timeout_depois_de_incumbent(self):
        instancia = instancia_pequena()
        ticks = iter(i * 0.05 for i in range(100))
        with patch("src.otimizacao.exato.monotonic", side_effect=lambda: next(ticks)):
            resultado = resolver_exato(instancia, 2, tempo_limite=0.27,
                                       max_candidatos=None, max_combinacoes=None)
        self.assertEqual(resultado.status, "limite")
        self.assertIsNotNone(resultado.alocacao)
        self.assertIsNone(resultado.bound)
        self.assertIsNone(resultado.gap)
        self.assertIsNotNone(resultado.lower_bound)

    def test_otimo_bound_certificado_e_lb_relaxado(self):
        instancia = instancia_pequena(matriz=((0, 9, 9),) * 4)
        resultado = resolver_exato(instancia, 1)
        self.assertEqual(resultado.status, "otimo")
        self.assertEqual(resultado.bound, resultado.custo)
        self.assertEqual(resultado.gap, 0)
        self.assertEqual(resultado.lower_bound, 0)
        self.assertLess(resultado.lower_bound, resultado.custo)

    def test_validacao_instancia_antes_de_poda(self):
        invalida = replace(instancia_pequena(), matriz_distancias=((float("nan"),),))
        with self.assertRaises(ValueError):
            resolver_exato(invalida, 1, max_candidatos=0)

    def test_cli_recusa_instancia_grande_sem_traceback(self):
        instancia = instancia_pequena()
        candidatos = tuple(replace(c, id=f"C{i}") for i, c in enumerate(instancia.candidatos * 4))
        matriz = instancia.matriz_distancias * 4
        grande = replace(instancia, candidatos=candidatos, matriz_distancias=matriz)
        stdout = StringIO()
        with patch("brute_force.ler_instancia", return_value=grande), redirect_stdout(stdout):
            codigo = cli_main(["--instancia", "dummy", "--tipos-por-local", "1"])
        self.assertEqual(codigo, 2)
        self.assertIn('"status": "limite"', stdout.getvalue())

    def test_matriz_zeros(self):
        instancia = instancia_pequena(matriz=((0, 0, 0),) * 4)
        resultado = resolver_exato(instancia, 2)
        self.assertEqual(resultado.status, "otimo")
        self.assertEqual(resultado.custo, 0)
        self.assertEqual(resultado.bound, 0)

    def test_validacao_parametros(self):
        for modo in (0, 3):
            with self.assertRaises(ValueError):
                resolver_exato(instancia_pequena(), modo)
        for tempo in (0, -1, math.inf, math.nan):
            with self.assertRaises(ValueError):
                resolver_exato(instancia_pequena(), 1, tempo_limite=tempo)

    def test_exemplo_textual(self):
        instancia = ler_instancia(Path("data/exemplos/pequena.txt"))
        self.assertEqual(len(instancia.candidatos), 6)
        self.assertEqual(resolver_exato(instancia, 1).status, "otimo")
        self.assertEqual(resolver_exato(instancia, 2).status, "otimo")


if __name__ == "__main__":
    unittest.main()
