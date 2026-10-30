import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

from src.instancias import Candidato, Escola, Instancia, carregar_instancia, ler_instancia, salvar_instancia
from src.instancias.models import ErroInstancia
from src.instancias.numerico import matriz_distancias_jax, matriz_distancias_numpy


def _texto(cabecalho="NUM_CANDIDATOS: 1\nNUM_ESCOLAS: 1", unidade="km", matriz="1.25"):
    cabecalho_matriz = (
        f"# MATRIZ_DISTANCIAS (em {unidade})" if unidade is not None
        else "# MATRIZ_DISTANCIAS"
    )
    return (
        f"{cabecalho}\n# CANDIDATOS (id, CEP, tipo_prova, latitude, longitude)\n"
        "C-01, 00123, M1, -21.0, -43.0\n# ESCOLAS (id, CEP, latitude, longitude, num_salas, capacidades_salas)\n"
        "E-01, 00456, -21.1, -43.1, 1, 10\n"
        f"{cabecalho_matriz}\n{matriz}\n"
    )


class InstanciasIOTest(unittest.TestCase):
    def carregar_texto(self, texto):
        temporario = tempfile.TemporaryDirectory()
        self.addCleanup(temporario.cleanup)
        caminho = Path(temporario.name) / "instancia.txt"
        caminho.write_text(texto, encoding="utf-8")
        return carregar_instancia(caminho)

    def test_cabecalho_quantidades_na_mesma_linha_e_alias(self):
        instancia = self.carregar_texto(_texto("NUM_CANDIDATOS: 1 NUM_ESCOLAS: 1"))
        self.assertEqual(ler_instancia, carregar_instancia)
        self.assertEqual(instancia.candidatos[0].id, "C-01")

    def test_cabecalho_quantidades_em_linhas_separadas_e_ids_string(self):
        instancia = self.carregar_texto(_texto())
        self.assertEqual(instancia.candidatos[0].cep, "00123")
        self.assertEqual(instancia.escolas[0].cep, "00456")

    def test_unidade_graus_roundtrip(self):
        instancia = self.carregar_texto(_texto(unidade="graus", matriz="0.0125"))
        self.assertEqual(instancia.unidade_distancias, "graus")
        with tempfile.TemporaryDirectory() as diretorio:
            caminho = Path(diretorio) / "saida.txt"
            salvar_instancia(instancia, caminho)
            recarregada = carregar_instancia(caminho)
        self.assertEqual(recarregada.unidade_distancias, "graus")
        self.assertEqual(recarregada.matriz_distancias, instancia.matriz_distancias)

    def test_matriz_invalida(self):
        with self.assertRaises(ErroInstancia):
            self.carregar_texto(_texto(matriz="1, 2"))

    def test_arquivo_so_com_caminho_rejeitado(self):
        with self.assertRaisesRegex(ErroInstancia, "apenas um caminho"):
            self.carregar_texto("../instancia-real.txt\n")

    def test_texto_arbitrario_nao_e_classificado_como_caminho(self):
        with self.assertRaises(ErroInstancia) as contexto:
            self.carregar_texto("observação isolada\n")
        self.assertNotIn("apenas um caminho", str(contexto.exception))

    def test_declaracoes_de_quantidade_invalidas(self):
        for cabecalho in (
            "NUM_CANDIDATOS: -1\nNUM_ESCOLAS: 1",
            "NUM_CANDIDATOS: 1.5 NUM_ESCOLAS: 1",
            "NUM_CANDIDATOS: 1xyz NUM_ESCOLAS: 1",
            "NUM_CANDIDATOS: 1 lixo NUM_ESCOLAS: 1",
            "NUM_CANDIDATOS: 1 NUM_CANDIDATOS: 1 NUM_ESCOLAS: 1",
            "NUM_CANDIDATOS: 1\nNUM_CANDIDATOS: 1\nNUM_ESCOLAS: 1",
        ):
            with self.subTest(cabecalho=cabecalho), self.assertRaises(ErroInstancia):
                self.carregar_texto(_texto(cabecalho))

    def test_unidade_desconhecida_e_sem_anotacao(self):
        for unidade in ("desconhecida", None):
            with self.subTest(unidade=unidade):
                instancia = self.carregar_texto(_texto(unidade=unidade))
                self.assertEqual(instancia.unidade_distancias, "desconhecida")

    def test_dispositivo_invalido(self):
        instancia = Instancia(
            (Candidato("C", "001", "M1", -21.0, -43.0),),
            (Escola("E", "002", -21.1, -43.1, (10,)),),
            ((1.25,),),
        )
        with self.assertRaisesRegex(ValueError, "cpu.*gpu"):
            matriz_distancias_jax(instancia, "tpu")

    def test_gpu_indisponivel_nao_faz_fallback(self):
        instancia = Instancia(
            (Candidato("C", "001", "M1", -21.0, -43.0),),
            (Escola("E", "002", -21.1, -43.1, (10,)),),
            ((1.25,),),
        )
        jax_falso = ModuleType("jax")
        jax_falso.__path__ = []
        jax_falso.config = SimpleNamespace(update=lambda *_: None)
        jax_falso.devices = lambda plataforma: (_ for _ in ()).throw(RuntimeError("indisponível"))
        jax_numpy_falso = ModuleType("jax.numpy")
        jax_falso.numpy = jax_numpy_falso
        with patch.dict("sys.modules", {"jax": jax_falso, "jax.numpy": jax_numpy_falso}):
            with self.assertRaisesRegex(RuntimeError, "gpu.*nenhum fallback"):
                matriz_distancias_jax(instancia, "gpu")

    def test_numpy_float64_cpu(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest("NumPy não instalado")
        instancia = self.carregar_texto(_texto())
        matriz = matriz_distancias_numpy(instancia)
        self.assertEqual(matriz.dtype, np.float64)
        self.assertEqual(matriz.tolist(), [[1.25]])

    def test_jax_opcional_cpu(self):
        try:
            import jax
        except ImportError:
            self.skipTest("JAX opcional não instalado")
        instancia = self.carregar_texto(_texto())
        matriz = matriz_distancias_jax(instancia, "cpu")
        self.assertEqual(str(matriz.dtype), "float64")
        self.assertEqual(matriz.device.platform, "cpu")


if __name__ == "__main__":
    unittest.main()
