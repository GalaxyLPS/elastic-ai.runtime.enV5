#ifndef SEQUENTIAL_H
#define SEQUENTIAL_H
#include "Tensor.h"
#include "Layer.h"

void sequentialForward(layer_t *model, size_t numberOfLayers, tensor_t *input, tensor_t* output);

#endif //SEQUENTIAL_H
