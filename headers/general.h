#include "define.h"

double error(double calculated, double expected);
double neuron_output(double inputs[], Neuron *n);
void forward_layer(Layer *layer, double inputs[]);
void backward_output_layer(Layer *layer, double inputs[], double expected[], double learning_rate) ;
void backward_hidden_layer(Layer *layer, Layer *next_layer, double inputs[], double learning_rate);
void update_weight(Layer *layer,double inputs[], double learning_rate);