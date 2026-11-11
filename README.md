# DCC136 — Trabalho 2

Projeto em Python para distribuir candidatos de uma prova entre escolas e salas, reduzindo a distância percorrida.

Cada candidato recebe uma escola e uma sala. A capacidade das salas é respeitada, cada sala ocupada recebe um único tipo de prova e cada escola pode aplicar K=1 ou K=2 tipos.

## Código

- `src/instancias/`: leitura e representação das instâncias.
- `src/otimizacao/`: construtivos, busca local, busca tabu, multi-exchange e modelos de otimização.
- `main.py`: execução de uma solução heurística em uma instância.
- `ferramentas/experimento_parada_adaptativa.py`: execução da heurística multi-start com busca tabu e multi-exchange.
- `ferramentas/resolver_gurobi.py`: MIP compacto para K=1 ou K=2.
- `ferramentas/resolver_gurobi_benders.py`: decomposição de Benders para K=1.
- `ferramentas/resolver_gurobi_benders_k2.py`: variante experimental de Benders para K=2.
- `data/instancias/`: instâncias usadas nos experimentos.
- `tests/`: testes automatizados; `notebooks/`: visualização e análise.

## Heurística

A heurística constrói uma distribuição inicial com regret e tenta melhorá-la com busca local. A variante multi-start com tabu e multi-exchange foi a mais consistente nos testes realizados. Ela encontra soluções viáveis rapidamente, mas não certifica que sejam ótimas.

Instale as dependências comuns:

```sh
python -m pip install -r requirements.txt
```

Resolva uma instância com regret e VND:

```sh
python main.py --instancia data/instancias/Instancia1_G1.txt --tipos-por-local 2 --seed 42 --execucoes 1
```

Rode o método multi-start em todas as instâncias:

```sh
python ferramentas/experimento_parada_adaptativa.py --workers 10 --metodo multistart_tabu_multiexchange --todas-instancias --seed-inicial 42 --quantidade-seeds 2 --saida resultados/multistart_tabu
```

## Solver Gurobi

O Gurobi oferece uma busca exata. O MIP compacto está disponível para K=1 e K=2. A decomposição de Benders foi usada nos resultados finais de K=1; o MIP compacto foi escolhido para K=2. O código Benders para K=2 fica disponível como alternativa experimental.

Instale a dependência opcional e confira se a licença está configurada localmente:

```sh
python -m pip install -r requirements-gurobi.txt
python ferramentas/resolver_gurobi.py --verificar-licenca
```

Exemplo de MIP compacto para K=2:

```sh
python ferramentas/resolver_gurobi.py --instancia data/instancias/Instancia1_G1.txt --tipos-por-local 2 --time-limit 5000 --mip-gap 0 --mip-focus 2 --threads 8 --seed 1321 --warm-start auto --saida resultados/gurobi
```

Para K=1, use o resolvedor Benders:

```sh
python ferramentas/resolver_gurobi_benders.py --instancia data/instancias/Instancia1_G1.txt --time-limit 5000 --mip-gap 0 --mip-focus 2 --master-slice 15 --cortes-raiz-inicial --max-rodadas-raiz 3 --cortes-magnanti-wong-raiz --threads 8 --seed 1321 --saida resultados/gurobi_benders
```

A licença acadêmica do Gurobi precisa estar ativa na máquina que executa o solver. As credenciais não fazem parte do repositório. Os resultados e arquivos de execução são gravados em `resultados/`, diretório ignorado pelo Git.

## Resultados

| Solver | Execuções | Resultado |
|---|---:|---|
| Heurística multi-start, tabu e multi-exchange | 56 combinações (melhor entre duas seeds) | 49 tiveram solução viável; 7 ficaram sem solução válida nas duas seeds. |
| Gurobi | 56 combinações | 33 ótimos globais provados, 5 inviáveis e 18 encerradas por limite de tempo. |

A tabela final compara os custos encontrados pela heurística e pelo Gurobi para cada instância e K. Para cada combinação, a heurística mostra a melhor solução válida entre as seeds 42 e 43.

### Resultado por K

| K | Ótimos provados | Inviáveis | Limite de tempo |
|---:|---:|---:|---:|
| 1 | 17 | 4 | 7 |
| 2 | 16 | 1 | 11 |
| Total | 33 | 5 | 18 |

## Comparação dos resultados por instância

Δ = custo heurístico menos a referência Gurobi. Para um ótimo provado, a referência é o ótimo; para uma execução limitada por tempo, é o incumbente UB. Valores positivos indicam que o custo heurístico foi maior. “Sem solução” significa que nenhuma das duas seeds encontrou uma alocação válida; não é prova de inviabilidade.

| Instância | K | Melhor heurística (seed, tempo) | Resultado Gurobi | Δ H−G |
|---|---:|---:|---|---:|
| Instancia1_G1.txt | 1 | 27987,3577 km (seed 43, 75,96 s) | ÓTIMO 26564,9078 km (3060,39 s) | +1422,4499 km (+5,35%) |
| Instancia1_G1.txt | 2 | 26295,6181 km (seed 43, 60,78 s) | LIMITE: UB 22746,3321; LB 22707,778126 km; gap 0,1695% (5003,17 s) | +3549,286 km (+15,6%) |
| Instancia1_G2.txt | 1 | — sem solução viável | INVIÁVEL (0,75 s) | — |
| Instancia1_G2.txt | 2 | — sem solução viável | INVIÁVEL (127,23 s) | — |
| Instancia1_G3.txt | 1 | 1042,189645 graus (seed 43, 55,58 s) | LIMITE: UB 1020,58917; LB 1007,85402 graus; gap 1,2478% (5005,63 s) | +21,6005 graus (+2,12%) |
| Instancia1_G3.txt | 2 | — sem solução viável | ÓTIMO 972,685066 graus (773,58 s) | — |
| Instancia1_G4.txt | 1 | 12833,2598 km (seed 42, 43,44 s) | ÓTIMO 12291,757 km (922,36 s) | +541,5028 km (+4,41%) |
| Instancia1_G4.txt | 2 | 8392,1087 km (seed 43, 95,79 s) | LIMITE: UB 7710,8201; LB 7708,916643 km; gap 0,0247% (5000,13 s) | +681,2886 km (+8,84%) |
| Instancia1_G5.txt | 1 | 15546,2733 km (seed 42, 74,16 s) | LIMITE: UB 14455,3757; LB 11935,1658 km; gap 17,4344% (5000,3 s) | +1090,8976 km (+7,55%) |
| Instancia1_G5.txt | 2 | 10831,7662 km (seed 43, 101,35 s) | ÓTIMO 9989,1422 km (1432,71 s) | +842,624 km (+8,44%) |
| Instancia1_G6.txt | 1 | 17703,9433 km (seed 42, 35,67 s) | LIMITE: UB 15870,3742; LB 15675,216166 km; gap 1,2297% (5000,64 s) | +1833,5691 km (+11,55%) |
| Instancia1_G6.txt | 2 | 17069,9055 km (seed 42, 135,3 s) | LIMITE: UB 10373,6699; LB 10325,153109 km; gap 0,4677% (5000,01 s) | +6696,2356 km (+64,55%) |
| Instancia1_G7.txt | 1 | 23260,45 km (seed 42, 66,79 s) | ÓTIMO 20444,87 km (75,09 s) | +2815,58 km (+13,77%) |
| Instancia1_G7.txt | 2 | 16995,65 km (seed 43, 40,43 s) | ÓTIMO 15037,81 km (648 s) | +1957,84 km (+13,02%) |
| instancia1_G8.txt | 1 | 1257,8417 km (seed 42, 102,43 s) | ÓTIMO 1185,3001 km (854,77 s) | +72,5416 km (+6,12%) |
| instancia1_G8.txt | 2 | 914,8846 km (seed 43, 81,92 s) | ÓTIMO 839,9692 km (5,29 s) | +74,9154 km (+8,92%) |
| Instancia1_G9.txt | 1 | 8532,4453 km (seed 43, 80,57 s) | ÓTIMO 7247,0485 km (467,86 s) | +1285,3968 km (+17,74%) |
| Instancia1_G9.txt | 2 | 6495,1223 km (seed 43, 95,73 s) | ÓTIMO 5763,8485 km (165,91 s) | +731,2738 km (+12,69%) |
| Instancia1_G10.txt | 1 | 11578,9629 km (seed 43, 82,84 s) | ÓTIMO 10080,8166 km (4146,17 s) | +1498,1463 km (+14,86%) |
| Instancia1_G10.txt | 2 | 8266,953 km (seed 43, 55,66 s) | LIMITE: UB 6954,5132; LB 6914,391037 km; gap 0,5769% (5048,14 s) | +1312,4398 km (+18,87%) |
| Instancia1_G11.txt | 1 | 37395,486 km (seed 42, 59,27 s) | ÓTIMO 34967,186 km (20,06 s) | +2428,3 km (+6,94%) |
| Instancia1_G11.txt | 2 | 25323,7045 km (seed 42, 60,56 s) | ÓTIMO 24376,1522 km (104,42 s) | +947,5523 km (+3,89%) |
| Instancia1_G12.txt | 1 | 20648,001 km (seed 43, 48,47 s) | LIMITE: UB 19389,2134; LB 16199,2232 km; gap 16,4524% (5000,3 s) | +1258,7876 km (+6,49%) |
| Instancia1_G12.txt | 2 | 15701,7136 km (seed 43, 85,79 s) | LIMITE: UB 15172,2544; LB 15169,349688 km; gap 0,0191% (5000,05 s) | +529,4592 km (+3,49%) |
| instancia_1_G13.txt | 1 | 1391,2705 km (seed 43, 74,84 s) | ÓTIMO 1311,0295 km (564,24 s) | +80,241 km (+6,12%) |
| instancia_1_G13.txt | 2 | 1080,7402 km (seed 42, 52,82 s) | ÓTIMO 985,7568 km (29,23 s) | +94,9834 km (+9,64%) |
| instancia_1_G14.txt | 1 | 1482,5781 km (seed 42, 33,61 s) | ÓTIMO 1390,3109 km (227,34 s) | +92,2672 km (+6,64%) |
| instancia_1_G14.txt | 2 | 1192,1559 km (seed 43, 55,67 s) | ÓTIMO 1091,4928 km (3,35 s) | +100,6631 km (+9,22%) |
| Instancia2_G1.txt | 1 | 16132,8816 km (seed 42, 69,82 s) | LIMITE: UB 14119,4563; LB 13040,914294 km; gap 7,6387% (5000,16 s) | +2013,4253 km (+14,26%) |
| Instancia2_G1.txt | 2 | 11694,0493 km (seed 43, 45,55 s) | ÓTIMO 10706,0503 km (4783,87 s) | +987,999 km (+9,23%) |
| Instancia2_G2.txt | 1 | — sem solução viável | INVIÁVEL (0,68 s) | — |
| Instancia2_G2.txt | 2 | 910,9653 km (seed 42, 33,54 s) | ÓTIMO 904,1992 km (0,61 s) | +6,7661 km (+0,75%) |
| Instancia2_G3.txt | 1 | — sem solução viável | INVIÁVEL (1,71 s) | — |
| Instancia2_G3.txt | 2 | 652,220507 graus (seed 43, 91,04 s) | ÓTIMO 648,686572 graus (302,19 s) | +3,5339 graus (+0,54%) |
| Instancia2_G4.txt | 1 | 11724,9943 km (seed 43, 61,55 s) | ÓTIMO 9867,9197 km (98,37 s) | +1857,0746 km (+18,82%) |
| Instancia2_G4.txt | 2 | 8625,9855 km (seed 42, 50,73 s) | ÓTIMO 7721,1294 km (1332,21 s) | +904,8561 km (+11,72%) |
| Instancia2_G5.txt | 1 | — sem solução viável | INVIÁVEL (5,33 s) | — |
| Instancia2_G5.txt | 2 | 13574,8986 km (seed 42, 76,09 s) | LIMITE: UB 12220,1255; LB 12216,676207 km; gap 0,0282% (5000,79 s) | +1354,7731 km (+11,09%) |
| Instancia2_G6.txt | 1 | 28581,0709 km (seed 42, 35,12 s) | LIMITE: UB 27632,7791; LB 25907,9117 km; gap 6,2421% (5000,16 s) | +948,2918 km (+3,43%) |
| Instancia2_G6.txt | 2 | 23478,8926 km (seed 42, 60,77 s) | LIMITE: UB 21822,7105; LB 21821,376215 km; gap 0,0061% (5000,09 s) | +1656,1821 km (+7,59%) |
| Instancia2_G7.txt | 1 | — sem solução viável | LIMITE: UB 63254,67; LB 61898,47 km; gap 2,144% (5000,12 s) | — |
| Instancia2_G7.txt | 2 | 66899,4 km (seed 43, 650 s) | LIMITE: UB 59126,36; LB 58679,367677 km; gap 0,756% (5000,08 s) | +7773,04 km (+13,15%) |
| instancia2_G8.txt | 1 | 117,0835 km (seed 42, 47,72 s) | ÓTIMO 117,0835 km (0,78 s) | +0 km (+0%) |
| instancia2_G8.txt | 2 | 656,505 km (seed 43, 53,35 s) | ÓTIMO 107,4827 km (0,08 s) | +549,0223 km (+510,8%) |
| Instancia2_G9.txt | 1 | 14083,953 km (seed 43, 75,94 s) | ÓTIMO 13454,5863 km (107,35 s) | +629,3667 km (+4,68%) |
| Instancia2_G9.txt | 2 | 12129,7313 km (seed 42, 91,25 s) | ÓTIMO 11423,9184 km (1643,17 s) | +705,8129 km (+6,18%) |
| Instancia2_G10.txt | 1 | 14012,5955 km (seed 43, 124,95 s) | ÓTIMO 12247,4375 km (78,19 s) | +1765,158 km (+14,41%) |
| Instancia2_G10.txt | 2 | 10958,7373 km (seed 43, 76,2 s) | LIMITE: UB 8118,2812; LB 8070,45699 km; gap 0,5891% (5000,02 s) | +2840,4561 km (+34,99%) |
| Instancia2_G11.txt | 1 | 103051,2983 km (seed 42, 45,54 s) | ÓTIMO 102678,434 km (97,52 s) | +372,8643 km (+0,36%) |
| Instancia2_G11.txt | 2 | 102504,76 km (seed 42, 70,94 s) | LIMITE: UB 102151,2024; LB 102149,408905 km; gap 0,0018% (5000,05 s) | +353,5576 km (+0,35%) |
| Instancia2_G12.txt | 1 | 96322,2624 km (seed 43, 74,78 s) | ÓTIMO 94065,5085 km (1107,27 s) | +2256,7539 km (+2,4%) |
| Instancia2_G12.txt | 2 | 92097,4229 km (seed 42, 70,7 s) | LIMITE: UB 91330,5666; LB 91253,838702 km; gap 0,084% (5000,03 s) | +766,8563 km (+0,84%) |
| instancia_2_G13.txt | 1 | 1358,7279 km (seed 43, 73,69 s) | ÓTIMO 1233,6675 km (7,5 s) | +125,0604 km (+10,14%) |
| instancia_2_G13.txt | 2 | 1107,9786 km (seed 42, 61,53 s) | ÓTIMO 950,1576 km (2,5 s) | +157,821 km (+16,61%) |
| instancia_2_G14.txt | 1 | 2193,4319 km (seed 43, 77,85 s) | ÓTIMO 1953,0526 km (0,78 s) | +240,3793 km (+12,31%) |
| instancia_2_G14.txt | 2 | 1919,0895 km (seed 42, 46,55 s) | ÓTIMO 1850,4831 km (2,03 s) | +68,6064 km (+3,71%) |
