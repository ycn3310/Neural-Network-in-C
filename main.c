#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

#include "headers/activation.h"
#include "headers/define.h"
#include "headers/init.h"
#include "headers/general.h"
#include "headers/csv_reader.h"

#define learning_rate 0.001
#define EPOCHS 1000000



int main() {
    srand(time(NULL));

    FILE *error_file, *csv;
    error_file = fopen("plot/error_file.csv", "w");
    csv = fopen("plot/Iris.csv", "r");

    if(error_file == NULL || csv == NULL){
        printf("failed to open file");
        return 1;
    }

    printf("files opened\n");

    fprintf(error_file, "epochs,error\n");

    int lenght;
    double *training_inputs[4];
    char **training_expected;
    double training[150];

    for(int i=1; i<5; i++){
        rewind(csv);
        training_inputs[i-1] = read_double_column(csv, i, &lenght);
    }
    printf("inputs extracted\n");

    rewind(csv);
    training_expected = read_string_column(csv, 5, &lenght);
    
    printf("outputs extracted\n");

    for(int i=0; i<lenght; i++){
        if(!strcmp(training_expected[i], "Iris-setosa")){
            training[i]=0;
        } else if (!strcmp(training_expected[i], "Iris-versicolor")){
            training[i]=1;
        } else{
            training[i]=2;
        }
    }

    printf("names changed to numbers\n");

    int shuffled_index[150];
    shuffle_array(shuffled_index, lenght);

    printf("indecies shuffled\n");
   
    Layer layer1 ={.neuron_count=4};
    init_layer(&layer1, 1);
    Layer layer2 ={.neuron_count=10};
    init_layer(&layer2, 4);
    Layer layer3 ={.neuron_count=1};
    init_layer(&layer3, 4);

    Layer layers[]={layer1,layer2,layer3};
    Network network = {.layer_count=3, .layers=layers};

    printf("network created\n");

    time_t start = time(NULL);
    for(int epoch=0; epoch< EPOCHS; epoch++){
        double error_rate=0;
        double inputs[4];
        for(int i=0; i<lenght-20; i++){ 
            int index = shuffled_index[i];
            for(int j=0; j<4; j++){
                inputs[j] = training_inputs[j][index];
            }

            forward_layer(&network.layers[0], inputs);
            forward_layer(&network.layers[1], network.layers[0].activation_array);
            forward_layer(&network.layers[2], network.layers[1].activation_array);

            backward_output_layer(&network.layers[2], network.layers[1].activation_array, &training[index], learning_rate);
            backward_hidden_layer(&network.layers[1], &network.layers[2], network.layers[0].activation_array, learning_rate);
            backward_hidden_layer(&network.layers[0], &network.layers[1], inputs, learning_rate);

            update_weight(&network.layers[2], network.layers[1].activation_array,learning_rate);
            update_weight(&network.layers[1], network.layers[0].activation_array,learning_rate);
            update_weight(&network.layers[0], inputs,learning_rate);

    
            error_rate += error(network.layers[2].activation_array[0], training[index]);
        }
        fprintf(error_file,"%i,%f\n",epoch,error_rate/130.0);
        if(epoch % 10000 ==0){
            printf("epoch: %i | error: %f\n", epoch, error_rate/130.0);
        }
    }

    printf("training done\n");

    time_t end = time(NULL);
    for(int i=lenght-20; i<lenght; i++){
        int index = shuffled_index[i];
        double inputs[4];
        for(int i=0; i<4; i++){
            inputs[i] = training_inputs[i][index];
        }
        forward_layer(&network.layers[0], inputs);
        forward_layer(&network.layers[1], network.layers[0].activation_array);
        forward_layer(&network.layers[2], network.layers[1].activation_array);

        double error_rate = network.layers[2].activation_array[0] - training[index];
        printf("prediction: %f | training expected: %f | error: %f\n",network.layers[2].activation_array[0],training[index], error_rate);
    }
    printf("testing done\n");
    printf("time taken: %jds\n", (intmax_t)(end-start));

    printf("closing the program\n");
    destroy_network(&network);
    fclose(error_file);
    return 0;
}


