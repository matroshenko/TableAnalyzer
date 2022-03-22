#include "intervals_centers.h"

#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include <cassert>

using namespace tensorflow;

REGISTER_OP("IntervalsCenters")
    .Input("mask: int32")
    .Output("centers: int32")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
        return c->Vector(shape_inference::InferenceContext::kUnknownDim);
    });

REGISTER_KERNEL_BUILDER(Name("IntervalsCenters").Device(DEVICE_CPU), IntervalsCentersOp);

//////////////////////////////////////////////////////////////////////////////
// IntervalsCentersOp

void Compute(OpKernelContext* context)
{
    const Tensor& mask = context->input(0);
    OP_REQUIRES(context, TensorShapeUtils::IsVector(mask.shape()),
        errors::InvalidArgument("IntervalsCenters expects a 1-D vector."));
    
    const vector<int> centers = getIntervalsCenters(mask);

    // Create an output tensor
    TensorShape resultShape({centers.size()});
    Tensor* resultTensor = 0;

    OP_REQUIRES_OK(
        context, context->allocate_output(0, resultShape, &resultTensor));
    auto resultVector = resultTensor->vec<int>();
    for (int i = 0; i < resultVector.size(); ++i) {
        resultVector(i) = centers[i];
  }
}

vector<int> IntervalsCentersOp::getIntervalsCenters(const Tensor& input) const
{
    const auto inputVec = input.vec<int>();
    
    vector<int> result;
    int currentIntervalStart = -1;
    bool isInsideInterval = false;
    
    for(int i = 0; i < inputVec.size(); ++i) {
        if(inputVec(i) == 1) {
            if(!isInsideInterval) {
                currentIntervalStart = i;
                isInsideInterval = true;
            }
        } else {
            if(isInsideInterval) {
                result.push_back((currentIntervalStart + i) / 2);
                isInsideInterval = false;
            }
        }
    }
    if(isInsideInterval) {
        result.push_back((currentIntervalStart + inputVec.size()) / 2);
    }
    return result;
}