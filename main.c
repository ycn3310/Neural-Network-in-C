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
#define EPOCHS 274000



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
            training[i]=0.0;
        } else if (!strcmp(training_expected[i], "Iris-versicolor")){
            training[i]=0.5;
        } else{
            training[i]=1.0;
        }
    }

    printf("names changed to numbers\n");

    int shuffled_index[150];
    shuffle_array(shuffled_index, lenght);

    printf("indecies shuffled\n");
   
    Layer layer1 ={.neuron_count=4};
    init_layer(&layer1, 4);
    Layer layer2 ={.neuron_count=10};
    init_layer(&layer2, 4);
    Layer layer3 ={.neuron_count=3};
    init_layer(&layer3, 10);

    Layer layers[]={layer1,layer2,layer3};
    Network network = {.layer_count=3, .layers=layers};

    printf("network created\n");

    time_t start = time(NULL);
    for(int epoch=0; epoch< EPOCHS; epoch++){
        double error_rate=0, inputs[4];

        for(int i=0; i<lenght-20; i++){ 

            int index = shuffled_index[i];

            for(int j=0; j<4; j++) inputs[j] = training_inputs[j][index];

            forward_network(&network,inputs);
            backward_network(&network, inputs, &training[index], learning_rate);
            update_network(&network, inputs, learning_rate);
    
            error_rate += error(max(network.layers[2]), training[index]);
        }

        fprintf(error_file,"%i,%f\n",epoch,error_rate/130.0);
        if(epoch % 10000 ==0) printf("epoch: %i | error: %f\n", epoch, error_rate/130.0);
        
    }

    printf("training done\n");

    time_t end = time(NULL);
    for(int i=lenght-20; i<lenght; i++){
        int index = shuffled_index[i]; double inputs[4];

        for(int i=0; i<4; i++){inputs[i] = training_inputs[i][index];}

        forward_network(&network,inputs);

        double *active = network.layers[2].activation_array;
        double error_rate = error(max(network.layers[2]),training[index]);
         
        printf("prediction: %.2f %.2f %.2f | training expected: %.1f | error: %.3f%%\n",active[0],active[1],active[2],training[index],100*error_rate);
    }

    printf("testing done\ntime taken: %jds\nclosing the program\n",(intmax_t)(end-start));

    save_model("mark1.txt", network);
    destroy_network(&network);
    fclose(error_file);

    return 0;
}


