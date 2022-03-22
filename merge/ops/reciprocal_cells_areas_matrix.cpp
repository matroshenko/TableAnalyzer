#include "reciprocal_cells_areas_matrix.h"

#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "grid_structure.h"
#include <cassert>

using namespace tensorflow;
using namespace shape_inference;

REGISTER_OP("ReciprocalCellsAreasMatrix")
    .Input("height: int32")
    .Input("width: int32")
    .Input("h_positions: int32")
    .Input("v_positions: int32")
    .Output("matrix: float32")
    .SetShapeFn([](InferenceContext* c) {
        c->set_output(0, c->Matrix(InferenceContext::kUnknownDim, InferenceContext::kUnknownDim));
        return Status::OK();
    });

REGISTER_KERNEL_BUILDER(
    Name("ReciprocalCellsAreasMatrix").Device(DEVICE_CPU), ReciprocalCellsAreasMatrixOp);

////////////////////////////////////////////////////////////////////////////////////////////////
// ReciprocalCellsAreasMatrixOp

void ReciprocalCellsAreasMatrixOp::Compute(OpKernelContext* context)
{
    const int height = context->input(0).scalar<int>()(0);
    const int width = context->input(1).scalar<int>()(0);
    const Tensor& hPositions = context->input(2);
    const Tensor& vPositions = context->input(3);

    // Create an output tensor
    TensorShape resultShape({height, width});
    Tensor* resultTensor = 0;
    OP_REQUIRES_OK(
        context, context->allocate_output(0, resultShape, &resultTensor));

    const GridStructure grid = createGridStructure(height, width, hPositions, vPositions);

    for(int rowIndex = 0; rowIndex < grid.GetRowsCount(); ++rowIndex) {
        for(int colIndex = 0; colIndex < grid.GetColsCount(); ++colIndex) {
            const Rect cell = grid.GetCellRect(rowIndex, colIndex);
            if(cell.IsEmpty()) {
                continue;
            }
            const int area = cell.GetArea();
            setValue(cell, 1.0f / cell.GetArea(), *resultTensor);
        }
    }
}

GridStructure ReciprocalCellsAreasMatrixOp::createGridStructure(
    int height, int width, const Tensor& hPositions, const Tensor& vPositions) const
{
    return GridStructure(
        extractPositions(hPositions, height),
        extractPositions(vPositions, width)
    );
}

vector<int> ReciprocalCellsAreasMatrixOp::extractPositions(
    const Tensor& tensor, int dimensionSize) const
{
    const auto vec = tensor.vec<int>();

    vector<int> result;
    result.reserve(vec.size() + 1);
    for(int i = 0; i < vec.size(); ++i) {
        result.push_back(vec(i));
    }
    result.push_back(dimensionSize);
    return result;
}

void ReciprocalCellsAreasMatrixOp::setValue(
    const Rect& rect, float value, Tensor& result) const
{
    auto resultMatrix = result.matrix<float>();
    for(int i = rect.Top; i < rect.Bottom; ++i) {
        for(int j = rect.Left; j < rect.Right; ++j) {
            resultMatrix(i, j) = value;
        }
    }
}