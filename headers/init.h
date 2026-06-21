#include "define.h"

void init_weights(double weights[], int number_of_weights);
void init_layer(Layer *layer, int number_of_inputs);
void destroy_layer(Layer *layer);
void destroy_network(Network *network);