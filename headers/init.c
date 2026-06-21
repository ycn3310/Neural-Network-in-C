#include "define.h"
#include <stdlib.h>

void init_weights(double weights[], int number_of_weights){
    for(int i=0; i < number_of_weights; i++){
        weights[i] = 2.0 * rand() / RAND_MAX - 1.0;
    }
}

void init_layer(Layer *layer, int number_of_inputs){
    layer->neurons = malloc(sizeof(Neuron) * layer->neuron_count);
    if (layer->neurons == NULL) {
        perror("malloc neurons");
        exit(EXIT_FAILURE);
    }

    for(int i = 0; i< layer->neuron_count; i++){
        Neuron *actual_neuron = &layer->neurons[i];
        actual_neuron->bias = 2.0 * rand() / RAND_MAX -1.0;
        actual_neuron->input_count = number_of_inputs;

        double *weights = malloc(sizeof(double) * number_of_inputs);
        if (weights == NULL) {
            perror("malloc");
            exit(EXIT_FAILURE);
        }

        init_weights(weights, number_of_inputs);
        actual_neuron->weights = weights;
        actual_neuron->delta = 0.0;
    }

    double *activation = malloc(sizeof(double) * layer->neuron_count);
    if (activation == NULL) {
        perror("activation");
        exit(EXIT_FAILURE);
    }
    layer->activation_array = activation;
}

void destroy_layer(Layer *layer){
    for(int i = 0; i< layer->neuron_count; i++){
        free(layer->neurons[i].weights);
    }
    free(layer->neurons);
    free(layer->activation_array);
}

void destroy_network(Network *network){
    for(int i=0; i<network->layer_count; i++){
        destroy_layer(&network->layers[i]);
    }
}