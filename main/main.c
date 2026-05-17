#include <stdio.h>
#include <string.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "driver/gpio.h"
#include "driver/touch_pad.h"

#define BOARD_LED        GPIO_NUM_2 // GPIO2 é o LED embutido na maioria dos ESP32
#define SAMPLE_COUNT     40
#define SAMPLE_PERIOD_MS 50

static void status_blink(int count, int delay_ms)
{
    for (int i = 0; i < count; i++) {
        gpio_set_level(BOARD_LED, 1);
        vTaskDelay(pdMS_TO_TICKS(delay_ms));
        gpio_set_level(BOARD_LED, 0);
        vTaskDelay(pdMS_TO_TICKS(delay_ms));
    }
}

void app_main(void)
{
    uint16_t touch_val_9, touch_val_8;

    // Inicialização do Hardware
    gpio_reset_pin(BOARD_LED);
    gpio_set_direction(BOARD_LED, GPIO_MODE_OUTPUT);
    
    touch_pad_init();
    touch_pad_config(TOUCH_PAD_NUM9, 0); // GPIO32
    touch_pad_config(TOUCH_PAD_NUM8, 0); // GPIO33

    printf("\n=== DATASET COLLECTOR ===\n");
    printf("0: no_touch | 1: one_touch | 2: two_touch | 3: hold_touch\n\n");

    while (1) {
        int input = getchar();

        if (input >= '0' && input <= '3') {
            int current_label = input - '0';

            // Alerta de contagem regressiva
            status_blink(5, 100);

            // Sinaliza início da captura (LED fixo)
            gpio_set_level(BOARD_LED, 1);

            printf("%d", current_label);

            for (int i = 0; i < SAMPLE_COUNT; i++) {
                touch_pad_read(TOUCH_PAD_NUM9, &touch_val_9);
                touch_pad_read(TOUCH_PAD_NUM8, &touch_val_8);

                printf(",%d,%d", touch_val_9, touch_val_8);
                vTaskDelay(pdMS_TO_TICKS(SAMPLE_PERIOD_MS));
            }
            
            printf("\n");
            gpio_set_level(BOARD_LED, 0);

            // Sinaliza fim da captura
            status_blink(2, 300);
        }

        vTaskDelay(pdMS_TO_TICKS(10));
    }
}