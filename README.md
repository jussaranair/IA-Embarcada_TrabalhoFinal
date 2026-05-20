# IA-Embarcada_TrabalhoFinal
# TinyML Gesture Recognition on ESP32

Sistema de reconhecimento de gestos touch utilizando **ESP32 + TensorFlow Lite Micro (TinyML)**.

O projeto realiza:

* coleta de dados via sensores touch capacitivos do ESP32;
* geração de dataset supervisionado;
* extração de características estatísticas;
* treinamento de rede neural;
* quantização INT8;
* inferência embarcada em tempo real.

---

# Objetivo

Desenvolver um sistema embarcado capaz de reconhecer gestos touch utilizando aprendizado de máquina diretamente no ESP32, sem necessidade de conexão com nuvem ou processamento externo.

---

# Gestos Reconhecidos

| Classe | Descrição  |
| ------ | ---------- |
| 0      | no_touch   |
| 1      | one_touch  |
| 2      | two_touch  |
| 3      | hold_touch |

---

# Arquitetura do Projeto

```text
ESP32 -> Coleta dos sensores
        ↓
Dataset CSV
        ↓
Python / Colab
- Extração de features
- Treinamento
- Validação
- Quantização
        ↓
TensorFlow Lite Micro
        ↓
ESP32 executa inferência em tempo real
```

---

# Hardware Utilizado

* ESP32
* Sensores Touch GPIO32 e GPIO33
* LED onboard (GPIO2)

---

# Tecnologias Utilizadas

* ESP-IDF
* TensorFlow Lite Micro
* TensorFlow / Keras
* Python
* NumPy
* Pandas
* Scikit-Learn
* Google Colab

---

# Coleta de Dados

O ESP32 captura amostras dos sensores touch e gera linhas CSV contendo:

* rótulo da classe;
* sequência temporal das leituras dos sensores.

Exemplo:

```csv
1,820,810,830,805,840,800,...
```

Cada captura utiliza:

* 2 sensores;
* 40 amostras temporais;
* janela temporal supervisionada.

---

# Engenharia de Atributos

Ao invés de utilizar diretamente toda a série temporal, foram extraídas características estatísticas para reduzir o custo computacional.

Para cada sensor são calculados:

* valor mínimo;
* média;
* desvio padrão;
* diferença temporal.

Total:

* 8 features por amostra.

---

# Arquitetura da Rede Neural

```text
Entrada: 8 features

Dense(32) + ReLU
Dropout(0.1)

Dense(16) + ReLU
Dropout(0.1)

Dense(4) + Softmax
```

---

# Quantização INT8

O modelo é convertido para TensorFlow Lite e posteriormente quantizado para INT8.

Benefícios:

* menor uso de memória;
* inferência mais rápida;
* menor consumo energético;
* compatibilidade com microcontroladores.

---

# Inferência no ESP32

O ESP32 executa:

1. leitura contínua dos sensores;
2. buffer circular de amostras;
3. extração de features;
4. normalização;
5. quantização;
6. inferência TinyML;
7. filtro de votação para estabilização.

---

# Estrutura do Projeto

```text
project/
│
├── dataset/
│   └── dataset.csv
│
├── training/
│   ├── train.py
│   ├── validate.py
│   ├── quantize.py
│   └── scaler.pkl
│
├── model/
│   ├── gesture_model.tflite
│   └── gesture_model_int8.tflite
│
├── esp32/
│   ├── dataset_collector/
│   └── tinyml_inference/
│
└── README.md
```

---

# Pipeline de Execução

## 1. Coletar Dataset

Executar o firmware de coleta no ESP32 para gerar o dataset CSV.

---

## 2. Treinar o Modelo

```python
train(
    csv_path="dataset.csv",
    epochs=100
)
```

---

## 3. Validar o Modelo

```python
evaluate(
    model_path="gesture_model.tflite",
    dataset_path="dataset.csv"
)
```

---

## 4. Quantizar Modelo

```python
quantize(model)
```

---

## 5. Exportar para ESP32

Converter o modelo `.tflite` para array C e incluir no firmware TinyML.

---

# Resultados

O sistema realiza inferência embarcada em tempo real utilizando TensorFlow Lite Micro diretamente no ESP32.

Principais características:

* baixa latência;
* baixo consumo de memória;
* execução local;
* reconhecimento estável dos gestos.

---

# Possíveis Melhorias

* adicionar novos gestos;
* utilizar CNN/LSTM;
* aumentar o dataset;
* otimização energética;
* múltiplos sensores;
* interface gráfica.

---

# Autoras

Andreia Nunes
Jussara André
Projeto acadêmico de TinyML utilizando ESP32.
