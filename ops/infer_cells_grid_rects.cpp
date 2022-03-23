#include "infer_cells_grid_rects.h"

#include "cells_structure_builder.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include <cassert>

using namespace tensorflow;

REGISTER_OP("InferCellsGridRects")
    .Input("merge_right_mask: bool")
    .Input("merge_down_mask: bool")
    .Output("rects: int32")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
        c->set_output(0, c->Matrix(shape_inference::InferenceContext::kUnknownDim, 4));
        return Status::OK();
    });

REGISTER_KERNEL_BUILDER(Name("InferCellsGridRects").Device(DEVICE_CPU), InferCellsGridRectsOp);

//////////////////////////////////////////////////////////////////////////////
// InferCellsGridRectsOp

void InferCellsGridRectsOp::Compute(OpKernelContext* context)
{
    const Tensor& mergeRightMask = context->input(0);
    const Tensor& mergeDownMask = context->input(1);

    const CellsStructureBuilder builder(mergeRightMask, mergeDownMask);
    const vector<Rect> cellsGridRects = builder.Build();

    // Create an output tensor
    TensorShape resultShape({static_cast<int>(cellsGridRects.size()), 4});
    Tensor* resultTensor = 0;
    OP_REQUIRES_OK(
        context, context->allocate_output(0, resultShape, &resultTensor));
        
    auto resultMatrix = resultTensor->matrix<int>();
    for(int i = 0; i < cellsGridRects.size(); ++i) {
        const Rect& rect = cellsGridRects[i];
        resultMatrix(i, 0) = rect.Left;
        resultMatrix(i, 1) = rect.Top;
        resultMatrix(i, 2) = rect.Right;
        resultMatrix(i, 3) = rect.Bottom;
    }
}