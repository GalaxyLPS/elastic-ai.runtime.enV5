#include "Linear.h"
#include "Quantization.h"
#include "Tensor.h"
#include "Sequential.h"
#include "Relu.h"
#include "unity.h"

#include <stddef.h>

void testLinearReluFloat() {
    float weightData[] = {-1.f, 2.f, -3.f, 4.f, 5.f, 6.f};
    float weightGrads[] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    size_t weightDims[] = {2, 3};
    size_t weightNumberOfDims = 2;
    size_t weightOrderOfDims[] = {0, 1};

    quantization_t weightQ;
    initFloat32Quantization(&weightQ);
    quantization_t weightGradQ;
    initFloat32Quantization(&weightGradQ);

    parameter_t weights;
    setParameterValues(&weights, weightData, &weightQ, weightGrads, &weightGradQ, weightDims,
                       weightNumberOfDims, weightOrderOfDims, NULL);

    float biasData[] = {-1.f, 3.f};
    float biasGrads[] = {0.f, 0.f};
    size_t biasDims[] = {2, 1};
    size_t biasNumberOfDims = 2;
    size_t biasOrderOfDims[] = {0, 1};

    quantization_t biasQ;
    initFloat32Quantization(&biasQ);
    quantization_t biasGradQ;
    initFloat32Quantization(&biasGradQ);

    parameter_t bias;
    setParameterValues(&bias, biasData, &biasQ, biasGrads, &biasGradQ, biasDims, biasNumberOfDims,
                       biasOrderOfDims, NULL);

    linearConfig_t linearConfig;
    initLinearConfig(&linearConfig, FLOATLAYER, &weights, &bias);

    float inputData[] = {0.f, 1.f, 2.f};
    size_t inputDims[] = {1, 3};
    size_t inputNumberOfDims = 2;
    size_t inputOrderOfDims[] = {0, 1};
    quantization_t inputQ;
    initFloat32Quantization(&inputQ);

    tensor_t input;
    setTensorValues(&input, inputData, inputDims, inputNumberOfDims, inputOrderOfDims, &inputQ,
                    NULL);

    float outputData[2] = {0, 0};
    size_t outputDims[] = {1, 2};
    size_t outputNumberOfDims = 2;
    size_t outputOrderOfDims[] = {0, 1};
    quantization_t outputQ;
    initFloat32Quantization(&outputQ);

    tensor_t output;
    setTensorValues(&output, outputData, outputDims, outputNumberOfDims, outputOrderOfDims,
                    &outputQ, NULL);

    layer_t linear;
    initLinearLayer(&linear, &linearConfig);

    layer_t relu;
    initReluLayer(&relu);

    layer_t model[] = {linear, relu};

    sequentialForward(model, 2, &input, &output);

    //float expected[] = {-5.f, -4.f};

    float expected[] = {0.f, 20.f};

    TEST_ASSERT_EQUAL_FLOAT_ARRAY(expected, output.data, 2);
}

void setUp() {}
void tearDown() {}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(testLinearReluFloat);
    UNITY_END();
}
