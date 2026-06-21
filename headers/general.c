#include "define.h"
#include "activation.h"
#include <stdio.h>

double error(double calculated, double expected){
    return (expected - calculated)*(expected - calculated);
}

double neuron_output(double inputs[], Neuron *n){
    double sum = n->bias;

    for(int i=0; i<n->input_count; i++){
        sum += inputs[i] * n->weights[i];
    }

    return sigmoid(sum);
}

void forward_layer(Layer *layer, double inputs[]){
    for(int neuron = 0; neuron < layer->neuron_count; neuron++){
        layer->activation_array[neuron] = neuron_output(inputs, &layer->neurons[neuron]);
    }
}

void backward_output_layer(Layer *layer, double inputs[], double expected[], double learning_rate) {

    for (int neuron = 0; neuron < layer->neuron_count; neuron++){
        Neuron *current_neuron = &layer->neurons[neuron];
        double activation = layer->activation_array[neuron];

        current_neuron->delta = dsigmoid(activation) * (expected[neuron] - activation);
    }
}

void backward_hidden_layer(Layer *layer, Layer *next_layer, double inputs[], double learning_rate){
    double sum =0;
    for(int weight=0; weight<layer->neuron_count; weight++){
        Neuron *current_neuron = &layer->neurons[weight];
        double activation = layer->activation_array[weight];

        for(int neuron=0; neuron < next_layer->neuron_count; neuron++){
            sum += next_layer->neurons[neuron].delta * next_layer->neurons[neuron].weights[weight]; 
        }
        current_neuron->delta = dsigmoid(activation) * sum;
    }
}

void update_weight(Layer *layer,double inputs[], double learning_rate){
    for(int neuron=0; neuron <layer->neuron_count; neuron++){
        Neuron *current_neuron = &layer->neurons[neuron];
        for(int weight=0; weight < current_neuron->input_count; weight++){
            current_neuron->weights[weight] += learning_rate * current_neuron->delta * inputs[weight];
        }
        current_neuron->bias += learning_rate * current_neuron->delta;
    }
}

/*void train(Network *network, double training_inputs[], double training_expected[][INPUT_SIZE], double learning_rate){
    forward_layer(&network->layers[0], training_inputs[]);
    for(int layer=0; layer<network->layer_count; layer++){
        forward_layer(&network->layers[1], network->layers[layer].activation_array);
    }

    backward_output_layer(&network.layers[2], network.layers[1].activation_array, training_expected[i]              , learning_rate);
    backward_hidden_layer(&network.layers[1], &network.layers[2]                , network.layers[0].activation_array, learning_rate);
    backward_hidden_layer(&network.layers[0], &network.layers[1]                , training_inputs[i]                , learning_rate);

    update_weight(&network.layers[2], network.layers[1].activation_array,learning_rate);
    update_weight(&network.layers[1], network.layers[0].activation_array,learning_rate);
    update_weight(&network.layers[0], training_inputs[i],learning_rate);
}*/