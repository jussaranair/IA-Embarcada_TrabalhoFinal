#include <math.h>
#include <stdio.h>
#include <string.h>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include "driver/touch_pad.h"

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "constants.h"
#include "main_functions.h"
#include "model.h"
#include "output_handler.h"

// ======================================================
// TensorFlow Lite Micro
// ======================================================

namespace {

const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;

TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

constexpr int kTensorArenaSize = 45000;
uint8_t tensor_arena[kTensorArenaSize];

}  // namespace

// ======================================================
// Configurações
// ======================================================

// #define WINDOW_SIZE 40
#define WINDOW_SIZE 20

#define NUM_SENSORS 2
#define NUM_CLASSES 4
#define PREDICTION_HISTORY_SIZE 5

// ======================================================
// Buffers e estado
// ======================================================

// Histórico circular para debounce da saída
static int prediction_history[PREDICTION_HISTORY_SIZE] = {0};
static int history_index = 0;

// Buffer circular das leituras dos sensores
static int16_t buffer[WINDOW_SIZE][NUM_SENSORS];
static int buffer_index = 0;
static bool buffer_ready = false;

// Baselines de calibração
static uint16_t baseline1 = 0;
static uint16_t baseline2 = 0;

// ======================================================
// Parâmetros do scaler
// ======================================================

const float SCALER_MEAN[8] = {
    547.905000f,
    867.709000f,
    156.919326f,
    -186.150000f,
    612.320000f,
    860.173376f,
    129.229682f,
    214.775000f};

const float SCALER_STD[8] = {
    346.622036f,
    226.354668f,
    123.742293f,
    310.569360f,
    364.265944f,
    260.476783f,
    124.692775f,
    307.036813f};

// ======================================================
// Setup
// ======================================================

void setup() {
  // Inicialização do touch pad
  touch_pad_init();
  touch_pad_set_fsm_mode(TOUCH_FSM_MODE_TIMER);

  touch_pad_config(TOUCH_PAD_NUM9, 0);  // GPIO 32
  touch_pad_config(TOUCH_PAD_NUM8, 0);  // GPIO 33

  vTaskDelay(pdMS_TO_TICKS(100));

  // Leitura inicial para baseline
  touch_pad_read(TOUCH_PAD_NUM9, &baseline1);
  touch_pad_read(TOUCH_PAD_NUM8, &baseline2);

  memset(buffer, 0, sizeof(buffer));
  memset(prediction_history, 0, sizeof(prediction_history));

  // ====================================================
  // Inicialização do modelo
  // ====================================================

  model = tflite::GetModel(g_model);

  if (model->version() != TFLITE_SCHEMA_VERSION) {
    MicroPrintf("Model version mismatch");
    return;
  }

  static tflite::MicroMutableOpResolver<3> resolver;

  resolver.AddFullyConnected();
  resolver.AddSoftmax();
  resolver.AddRelu();

  static tflite::MicroInterpreter static_interpreter(
      model,
      resolver,
      tensor_arena,
      kTensorArenaSize);

  interpreter = &static_interpreter;

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    MicroPrintf("AllocateTensors failed");
    return;
  }

  input = interpreter->input(0);
  output = interpreter->output(0);

  MicroPrintf("Setup complete. Monitorando dados brutos...");
}

// ======================================================
// Inferência
// ======================================================

void run_inference() {
  float features[8] = {0.0f};

  for (int sensor = 0; sensor < NUM_SENSORS; sensor++) {
    float min_val = static_cast<float>(buffer[0][sensor]);
    float max_val = static_cast<float>(buffer[0][sensor]);
    float sum_val = 0.0f;

    float primeiro_do_buffer = 0.0f;
    float ultimo_do_buffer = 0.0f;

    // Reconstrói a ordem cronológica correta do buffer circular
    for (int i = 0; i < WINDOW_SIZE; i++) {
      int cronologico_idx = (buffer_index + i) % WINDOW_SIZE;

      float val = static_cast<float>(buffer[cronologico_idx][sensor]);

      if (i == 0) {
        primeiro_do_buffer = val;
      }

      if (i == WINDOW_SIZE - 1) {
        ultimo_do_buffer = val;
      }

      if (val < min_val) {
        min_val = val;
      }

      if (val > max_val) {
        max_val = val;
      }

      sum_val += val;
    }

    float mean_val = sum_val / static_cast<float>(WINDOW_SIZE);

    float variance_sum = 0.0f;

    for (int i = 0; i < WINDOW_SIZE; i++) {
      float diff =
          static_cast<float>(buffer[i][sensor]) - mean_val;

      variance_sum += diff * diff;
    }

    float std_val =
        sqrtf(variance_sum / static_cast<float>(WINDOW_SIZE));

    // Diferença temporal da janela
    float diff_val = ultimo_do_buffer - primeiro_do_buffer;

    int offset = sensor * 4;

    features[offset + 0] = min_val;
    features[offset + 1] = mean_val;
    features[offset + 2] = std_val;
    features[offset + 3] = diff_val;
  }

  // ====================================================
  // Normalização + quantização INT8
  // ====================================================

  float scale = input->params.scale;
  int zero_point = input->params.zero_point;

  for (int i = 0; i < 8; i++) {
    float norm_x =
        (features[i] - SCALER_MEAN[i]) / SCALER_STD[i];

    int32_t quantized_val =
        static_cast<int32_t>(roundf(norm_x / scale)) +
        zero_point;

    if (quantized_val < -128) {
      quantized_val = -128;
    }

    if (quantized_val > 127) {
      quantized_val = 127;
    }

    input->data.int8[i] =
        static_cast<int8_t>(quantized_val);
  }

  if (interpreter->Invoke() != kTfLiteOk) {
    MicroPrintf("Invoke failed");
    return;
  }

  // ====================================================
  // Argmax
  // ====================================================

  int best_class = 0;
  int8_t best_score = output->data.int8[0];

  for (int i = 1; i < NUM_CLASSES; i++) {
    if (output->data.int8[i] > best_score) {
      best_score = output->data.int8[i];
      best_class = i;
    }
  }

  // ====================================================
  // Filtro de votação
  // ====================================================

  prediction_history[history_index] = best_class;

  history_index =
      (history_index + 1) % PREDICTION_HISTORY_SIZE;

  int class_counts[NUM_CLASSES] = {0};

  for (int i = 0; i < PREDICTION_HISTORY_SIZE; i++) {
    class_counts[prediction_history[i]]++;
  }

  int classe_estabilizada = 0;
  int max_votos = 0;

  for (int i = 0; i < NUM_CLASSES; i++) {
    if (class_counts[i] > max_votos) {
      max_votos = class_counts[i];
      classe_estabilizada = i;
    }
  }

  static int ultima_classe_printada = -1;

  if (max_votos >= 4 &&
      classe_estabilizada != ultima_classe_printada) {
    ultima_classe_printada = classe_estabilizada;

    const char* labels[NUM_CLASSES] = {
        "no_touch",
        "one_touch",
        "two_touch",
        "hold_touch"};

    MicroPrintf(
        "Gesto Confirmado: %s",
        labels[classe_estabilizada]);
  }
}

// ======================================================
// Loop principal
// ======================================================

void loop() {
  uint16_t v1 = 0;
  uint16_t v2 = 0;

  touch_pad_read(TOUCH_PAD_NUM9, &v1);

  esp_rom_delay_us(50);

  touch_pad_read(TOUCH_PAD_NUM8, &v2);

  int d1 = abs(static_cast<int>(v1));
  int d2 = abs(static_cast<int>(v2));

  buffer[buffer_index][0] = d1;
  buffer[buffer_index][1] = d2;

  buffer_index++;

  if (buffer_index >= WINDOW_SIZE) {
    buffer_index = 0;
    buffer_ready = true;
  }

  // Executa inferência a cada 3 amostras (~150 ms)
  if (buffer_ready && (buffer_index % 3 == 0)) {
    run_inference();
  }

  vTaskDelay(pdMS_TO_TICKS(50));
}