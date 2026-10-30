# DCC136 — Trabalho 1

Implementação em Python para alocar candidatos a salas de prova, usando a matriz de distâncias fornecida. Cada candidato recebe uma escola e uma sala; a capacidade é respeitada e cada sala ocupada aplica um único tipo de prova.

## Método

A solução usa construção regret aleatorizada, múltiplas partidas, busca tabu por deslocamentos e trocas, e ciclos multi-exchange entre escolas que atendem o mesmo tipo. O limite K é configurável: K=1 ou K=2 tipos por escola. Toda solução registrada passa pelo verificador de salas.

## Resultados

Foram executadas 112 combinações: 28 instâncias, K=1 e K=2, duas seeds (42 e 43), com dez processos. O método parou após 30 s sem melhora e teve teto de 650 s por execução. Houve 98 soluções válidas e 14 execuções sem solução válida dentro do teto; isso não prova inviabilidade. Em 41 comparações pareadas com a planilha, nenhuma solução venceu o melhor resultado publicado.

Os custos mantêm a unidade de cada instância e não devem ser comparados entre unidades diferentes. Os arquivos brutos por seed permanecem no diretório local de resultados e não fazem parte do histórico versionado.

## Executar

Instale os pacotes e rode os testes:

```sh
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

Para repetir o lote do método escolhido:

```sh
python ferramentas/experimento_parada_adaptativa.py \
  --workers 10 --metodo multistart_tabu_multiexchange \
  --todas-instancias --seed-inicial 42 --quantidade-seeds 2 \
  --saida resultados/lote_multiexchange_novo
```
