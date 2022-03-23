#pragma once

#include <vector>
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"

using namespace tensorflow;
using std::vector;

class GridStructure;
struct Rect;

class IndicesCubeOp : public OpKernel {
public:
    explicit IndicesCubeOp(OpKernelConstruction* context) : OpKernel(context) {}

    virtual void Compute(OpKernelContext* context) override;

private:
    GridStructure createGridStructure(
        int height, int width, const Tensor& hPositions, const Tensor& vPositions) const;
    vector<int> extractPositions(const Tensor& tensor, int dimensionSize) const;
    void setValues(const Rect& rect, int value1, int value2, Tensor& result) const;
};