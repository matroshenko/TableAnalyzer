#pragma once

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"

using namespace tensorflow;

class InferCellsGridRectsOp : public OpKernel {
public:
    explicit InferCellsGridRectsOp(OpKernelConstruction* context) : OpKernel(context) {}

    virtual void Compute(OpKernelContext* context) override;
};