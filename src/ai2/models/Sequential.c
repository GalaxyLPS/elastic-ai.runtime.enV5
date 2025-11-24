#include "Sequential.h"
#include "Layer.h"
#include "TensorConversion.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// TODO FIX
void sequentialForward(layer_t *model,
                       size_t numberOfLayers,
                       tensor_t *input,
                       tensor_t *output)
{
    size_t bytesPerElement = calcBytesPerElement(input->quantization);

    for (size_t i = 0; i < numberOfLayers; i++) {

        layer_t *currentLayer = &model[i];

        size_t outputNumberOfDims = input->shape.numberOfDimensions;

        size_t *outDims = calloc(outputNumberOfDims, sizeof(size_t));
        size_t *outOrder = calloc(outputNumberOfDims, sizeof(size_t));

        shape_t outShape = {
            .dimensions = outDims,
            .numberOfDimensions = outputNumberOfDims,
            .orderOfDimensions = outOrder
        };

        currentLayer->calcOutputShape(currentLayer,
                                      &input->shape,
                                      &outShape);

        size_t numValues =
            calcNumberOfElementsByDims(outShape.numberOfDimensions,
                                       outShape.dimensions);

        /* allocate persistent memory for intermediate result */
        uint8_t *data = malloc(numValues * bytesPerElement);

        tensor_t intermediateOutput;
        setTensorValues(&intermediateOutput,
                        data,
                        outShape.dimensions,
                        outShape.numberOfDimensions,
                        outShape.orderOfDimensions,
                        input->quantization,
                        NULL);

        currentLayer->forward(currentLayer->layerConfig,
                              input,
                              &intermediateOutput);

        float *test = (float *)intermediateOutput.data;
        for(size_t j = 0; j < 2; j++) {
            printf("x: %f\n", test[j]);
        }

        memcpy(input->data,
               intermediateOutput.data,
               numValues * bytesPerElement);

        memcpy(input->shape.dimensions,
               intermediateOutput.shape.dimensions,
               outputNumberOfDims * sizeof(size_t));

        memcpy(input->shape.orderOfDimensions,
               intermediateOutput.shape.orderOfDimensions,
               outputNumberOfDims * sizeof(size_t));

        input->shape.numberOfDimensions = intermediateOutput.shape.numberOfDimensions;

        free(outDims);
        free(outOrder);
        free(data);
    }
    float *test = (float *)input->data;
    for(size_t i = 0; i < 2; i++) {
        printf("x: %f\n", test[i]);
    }
    convertTensor(input, output);
}


/*loss *getLossFunctionByType(lossFunctionType_t lossType) {
    loss *lossFunction = calloc(1, sizeof(loss));
    switch (lossType) {
    case MSE:
        lossFunction = MSELossBackward;
        break;
    case CROSS_ENTROPY:
        lossFunction = crossEntropySoftmaxBackward;
        break;
    default:
        printf("Loss type not found");
        break;
    }
    return lossFunction;
}

/*! IMPORTANT: We assume, that if you use Cross Entropy as your loss function,
 * you also use Softmax with it. We introduce Softmax as a dedicated Layer,
 * but in the backward pass it is ignored. We do this, because the Cross Entropy Backward
 * already takes the Softmax Backward into account.
 #1#
trainingStats_t *sequentialCalculateGrads(layer_t *model,
                                          size_t sizeNetwork,
                                          lossFunctionType_t lossFunctionType,
                                          tensor_t *input,
                                          tensor_t *label) {

    tensor_t *layerOutputs[sizeNetwork + 1];
    layerOutputs[0] = input;

    // Forward Pass
    for (size_t i = 0; i < sizeNetwork; i++) {
        layer_t currentLayer = model[i];

        forwardFn_t fwd = currentLayer.forward;

        layerOutputs[i + 1] = fwd(currentLayer->config, layerOutputs[i]);
    }

    // Loss
    loss *lossFunction = getLossFunctionByType(lossFunctionType);
    // TODO adapt loss functions
    qTensor_t *gradQTensor = lossFunction(layerOutputs[sizeNetwork], labelQTensor);
    trainingStats_t *trainingStats = calloc(1, sizeof(trainingStats_t));
    // TODO adapt trainingStats_t
    trainingStats->loss = gradQTensor->data;

    size_t backwardIndex = sizeNetwork - 1;
    if (lossFunctionType == CROSS_ENTROPY) {
        backwardIndex -= 1;
    }
    // Backward Pass
    for (int i = (int)(backwardIndex); i >= 0; i--) {
        backward *bwd = layerFunctions[network[i]->type].backwardFunc;
        gradQTensor = bwd(network[i]->config, gradQTensor, layerOutputs[i]);
    }


    trainingStats->output = gradQTensor->data;

    return trainingStats;
}*/
