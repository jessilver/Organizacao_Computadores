# Simulador de Processador RISC-V com Pipeline e Cache

Este projeto consiste em um simulador interativo de um processador RISC-V simplificado, desenvolvido em Python utilizando o ambiente Google Colab. A ferramenta permite a análise e comparação de diferentes arquiteturas e otimizações, como a execução em ciclo único vs. pipeline, tratamento de hazards e o impacto de uma hierarquia de memória com cache.

O simulador foi desenvolvido como parte das atividades acadêmicas das disciplinas de Organização e Arquitetura de Computadores da UFV, baseando-se nos trabalhos e materiais propostos pelo Professor Ricardo Ferreira.

## ✨ Funcionalidades Principais

O simulador implementa um vasto conjunto de características de processadores modernos:

- **Dois Modos de Execução:**
  - **Ciclo Único:** Execução sequencial, instrução por instrução, com controle interativo de passo a passo.
  - **Pipeline Completo:** Simulação de um pipeline de 4 estágios (IF, ID, EX, MEM_WB).

- **Conjunto de Instruções Abrangente (Parte 1 & 2):**
  - **Aritméticas de Inteiros:** `ADD`, `SUB`, `MUL`, `DIV`, `ADDI` com detecção de overflow.
  - **Aritméticas de Ponto Flutuante:** `ADD.S`, `MUL.S` (precisão simples IEEE 754).
  - **Acesso à Memória:** `LW` (Load), `SW` (Store).
  - **Desvios (Branches):** `BEQ` (Branch if Equal), `J` (Jump incondicional) com suporte a labels.

- **Análise de Pipeline (Parte 3 & 4):**
  - Visualização da evolução do pipeline em uma tabela `tempo × estágio`.
  - Detecção e log detalhado de **Data Hazards** (EX->ID e Load-Use) e **Control Hazards** (BEQ, J).
  - Implementação e comparação de três técnicas de tratamento de hazards:
    1.  Nenhum tratamento (para observar os resultados incorretos).
    2.  **Stall** (inserção de bolhas).
    3.  **Forwarding** (bypass).

- **Hierarquia de Memória (Parte 5):**
  - Simulação de uma **cache associativa por conjunto de 2 vias**.
  - Cálculo e exibição de estatísticas de cache: hits, misses, e taxas de acerto/falha.
  - Simulação de **penalidade de miss** configurável.
  - Comparação de políticas de substituição: **LRU** (Least Recently Used) vs. **Random**.

- **Análise de Desempenho:**
  - Medição e comparação do número total de ciclos entre os diferentes modos de execução.
  - Cálculo automático do **speedup** obtido com as otimizações de pipeline e cache em relação ao ciclo único.

- **Interface Interativa:**
  - Desenvolvida com `ipywidgets` para uma experiência de usuário amigável, com seletores dinâmicos e botões de controle.

## 🚀 Como Executar

A forma mais fácil de executar o simulador é abrindo o notebook diretamente no Google Colab.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jessilver/Organizacao_Computadores/blob/develop/Template.ipynb)

1.  Clique no botão "Open In Colab" acima.
2.  No notebook aberto, vá para o menu **`Ambiente de execução`** e clique em **`Executar tudo`**.
3.  Role até a célula de execução para interagir com a interface do simulador.
4.  Existem intruções de prontas para testes preparadas mais a baixo no projeto.

## 🕹️ Visão Geral da Interface

A interface permite controlar todos os aspectos da simulação:

- **Modo de Simulação:** Alterne entre `Ciclo Único` e `Pipeline (Completo)`.
- **Opções do Pipeline:** Ficam visíveis apenas no modo Pipeline.
  - **Tratamento de Hazard:** Escolha entre `Sem tratamento`, `Stall` ou `Forwarding`.
  - **Hierarquia de Memória:** Habilite ou desabilite a `Cache`.
  - **Política de Cache:** Escolha entre `LRU` e `Random` (visível apenas com a cache ativa).
  - **Penalidade de Miss:** Defina o número de ciclos de penalidade para um cache miss.
- **Área de Código:** Onde você escreve seu programa em Assembly RISC-V.
- **Botões de Controle:**
  - **`Load/Reset`**: Carrega o código e reinicia o estado do processador.
  - **`Próximo Passo ->` / `Próximo Ciclo ->`**: Executa uma instrução (Ciclo Único) ou um ciclo de clock (Pipeline).
  - **`Run All`**: Executa o programa inteiro de uma vez.
  - **`Show Registers` / `Show Memory` / `Show Cache Stats`**: Exibe o estado atual do processador e da memória.

## 🔬 Experimentos e Análises

O notebook contém um bloco final (**Bloco 10**) dedicado a uma série de experimentos pré-definidos. Este bloco serve como um guia para validar cada parte do projeto e analisar os resultados, com códigos de teste prontos para serem copiados e executados.

Os experimentos cobrem desde a validação de operações básicas até a medição do impacto de hazards, da taxa de acerto da cache e o cálculo do speedup final.

## 👨‍💻 Autor

- **[jessilver](https://github.com/jessilver)**

---
*Este trabalho é uma derivação e junção dos trabalhos das disciplinas de Organização de Computadores e Arquitetura de Computadores do Professor Ricardo Ferreira da UFV.*
