#include "define.h"
#include "activation.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

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

void shuffle_array(int array[], int n){
    for(int i=0; i<n; i++){
        array[i]=i;
    }
    for(int i =n-1; i>0; i--){
        int j = rand() % (i+1);

        int temp = array[i];
        array[i] = array[j];
        array[j] = temp;
    }
}

void softmax(Layer *layer, double t){
    double sum = 0;
    for(int i =0; i < layer->neuron_count; i++){
        sum += exp(layer->activation_array[i]/t);
    }

    for(int i=0; i< layer->neuron_count; i++){
        layer->activation_array[i] = exp(layer->activation_array[i]/t)/sum;
    }
}

double max(Layer layer){
    double maximum = layer.activation_array[0];
    for(int i=0; i<layer.neuron_count; i++){
        if(layer.activation_array[i] > maximum) maximum = layer.activation_array[i];
    }
    return maximum;
}

void save_model(char *destination, Network network){
    FILE *model;
    model = fopen(destination, "w");

    for(int layer =0; layer < network.layer_count; layer++){
        fprintf(model,"layer %i:\n", layer);
        for(int neuron =0; neuron < network.layers[layer].neuron_count; neuron++){
            fprintf(model,"neuron %i: ",neuron);
            for(int weight=0; weight < network.layers[layer].neurons[neuron].input_count; weight++){
                fprintf(model, "%f ", network.layers[layer].neurons[neuron].weights[weight]);
            }
            fprintf(model,"\n");
        }
        fprintf(model,"\n");
    }

    fclose(model);
}

void forward_network(Network *network, double inputs[]){
    forward_layer(&network->layers[0], inputs);
    for(int i=1; i<network->layer_count; i++){
        forward_layer(&network->layers[i], network->layers[i-1].activation_array);
    }
}

void update_network(Network *network, double inputs[], double learning_rate){
    for(int i=network->layer_count-1; i>1; i--){
        update_weight(&network->layers[i], network->layers[i-1].activation_array,learning_rate);
    }
    update_weight(&network->layers[1], inputs,learning_rate);
}


void backward_network(Network *network, double inputs[], double expected[], double learning_rate){
    backward_output_layer(&network->layers[network->layer_count-1], network->layers[network->layer_count-2].activation_array, expected, learning_rate);
    for(int i=network->layer_count-2; i>1; i--){
        backward_hidden_layer(&network->layers[i], &network->layers[i+1], network->layers[i-1].activation_array, learning_rate);
    }
    backward_hidden_layer(&network->layers[1], &network->layers[2], inputs, learning_rate);
}



