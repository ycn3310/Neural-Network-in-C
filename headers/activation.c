#include <math.h>

double sigmoid(double z){
    return 1.0/(1.0+exp(-z));
}

double dsigmoid(double z){
    return z * (1 - z);
}

double delta(double prediction, double excepected){
    return (prediction - excepected) * dsigmoid(prediction);
}

double Relu(double z){
    if(z<0){
        return 0.0;
    } else {
        return z;
    }
}