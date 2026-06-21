#ifndef DEFINE_H
#define DEFINE_H

#define NUM_EXAMPLES 4
#define INPUT_SIZE 2
#define OUTPUT_SIZE 1

typedef struct{
    double *weights;
    int input_count;
    double bias;
    double delta;
} Neuron;

typedef struct{
    Neuron *neurons;
    double *activation_array;
    int neuron_count;
} Layer;

typedef struct {
    Layer *layers;
    int layer_count;
} Network;

#endif