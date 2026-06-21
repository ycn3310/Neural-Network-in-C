#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>

#include "headers/activation.h"
#include "headers/define.h"
#include "headers/init.h"
#include "headers/general.h"

#define learning_rate 0.1
#define EPOCHS 30000



int main() {
    srand(time(NULL));

    FILE *error_file;
    error_file = fopen("plot/error_file.csv", "w");
    if(error_file == NULL){
        printf("failed to open file");
        return 1;
    }
    fprintf(error_file, "epochs,error\n");

    double training_inputs[NUM_EXAMPLES][INPUT_SIZE] = {
        {0, 0},
        {0, 1},
        {1, 0},
        {1, 1}
    };

    double training_expected[NUM_EXAMPLES][OUTPUT_SIZE] = {
        {0},
        {1},
        {1},
        {1}
    };


   
    Layer layer1 ={.neuron_count=3};
    init_layer(&layer1, 2);
    Layer layer2 ={.neuron_count=2};
    init_layer(&layer2, 3);
    Layer layer3 ={.neuron_count=1};
    init_layer(&layer3, 2);

    Layer layers[]={layer1,layer2,layer3};
    Network network = {.layer_count=3, .layers=layers};

    time_t start = time(NULL);
    for(int epoch=0; epoch< EPOCHS; epoch++){
        double error_rate=0;
        for(int i=0; i<4; i++){ 
            forward_layer(&network.layers[0], training_inputs[i]);
            forward_layer(&network.layers[1], network.layers[0].activation_array);
            forward_layer(&network.layers[2], network.layers[1].activation_array);

            backward_output_layer(&network.layers[2], network.layers[1].activation_array, training_expected[i], learning_rate);
            backward_hidden_layer(&network.layers[1], &network.layers[2], network.layers[0].activation_array, learning_rate);
            backward_hidden_layer(&network.layers[0], &network.layers[1], training_inputs[i], learning_rate);

            update_weight(&network.layers[2], network.layers[1].activation_array,learning_rate);
            update_weight(&network.layers[1], network.layers[0].activation_array,learning_rate);
            update_weight(&network.layers[0], training_inputs[i],learning_rate);

            error_rate += error(network.layers[2].activation_array[0], training_expected[i][0]);
        }
        fprintf(error_file,"%i,%f\n",epoch,error_rate/4.0);
    }
    time_t end = time(NULL);
    for(int i=0; i<4; i++){
        forward_layer(&network.layers[0], training_inputs[i]);
        forward_layer(&network.layers[1], layer1.activation_array);
        forward_layer(&network.layers[2], layer2.activation_array);

        double error_rate = network.layers[2].activation_array[0] - training_expected[i][0];
        printf("prediction: %f | training expected: %f | error: %f\n",network.layers[2].activation_array[0],training_expected[i][0], error_rate);
    }
    printf("time taken: %jd", (intmax_t)(end-start));


    destroy_network(&network);
    fclose(error_file);
    return 0;
}


