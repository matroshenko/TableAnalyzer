#pragma once

#include <vector>
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"

using namespace tensorflow;
using std::vector;

class IntervalsCentersOp : public OpKernel {
public:
    explicit IntervalsCentersOp(OpKernelConstruction* context) : OpKernel(context) {}

    virtual void Compute(OpKernelContext* context) override;

private:
    vector<int> getIntervalsCenters(const Tensor& input) const;
};