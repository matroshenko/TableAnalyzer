#include "indices_cube.h"

#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "grid_structure.h"
#include <cassert>

using namespace tensorflow;
using namespace shape_inference;

REGISTER_OP("IndicesCube")
    .Input("height: int32")
    .Input("width: int32")
    .Input("h_positions: int32")
    .Input("v_positions: int32")
    .Output("cube: int32")
    .SetShapeFn([](InferenceContext* c) {
        ShapeHandle shape = c->MakeShape(
            {InferenceContext::kUnknownDim, InferenceContext::kUnknownDim, 2});
        c->set_output(0, shape);
        return Status::OK();
    });

REGISTER_KERNEL_BUILDER(
    Name("IndicesCube").Device(DEVICE_CPU), IndicesCubeOp);

////////////////////////////////////////////////////////////////////////////////////////////////
// IndicesCubeOp

void IndicesCubeOp::Compute(OpKernelContext* context)
{
    const int height = context->input(0).scalar<int>()(0);
    const int width = context->input(1).scalar<int>()(0);
    const Tensor& hPositions = context->input(2);
    const Tensor& vPositions = context->input(3);

    // Create an output tensor
    TensorShape resultShape({height, width, 2});
    Tensor* resultTensor = 0;
    OP_REQUIRES_OK(
        context, context->allocate_output(0, resultShape, &resultTensor));

    const GridStructure grid = createGridStructure(height, width, hPositions, vPositions);

    for(int rowIndex = 0; rowIndex < grid.GetRowsCount(); ++rowIndex) {
        for(int colIndex = 0; colIndex < grid.GetColsCount(); ++colIndex) {
            const Rect cell = grid.GetCellRect(rowIndex, colIndex);
            setValues(cell, rowIndex, colIndex, *resultTensor);
        }
    }
}

GridStructure IndicesCubeOp::createGridStructure(
    int height, int width, const Tensor& hPositions, const Tensor& vPositions) const
{
    return GridStructure(
        extractPositions(hPositions, height),
        extractPositions(vPositions, width)
    );
}

vector<int> IndicesCubeOp::extractPositions(
    const Tensor& tensor, int dimensionSize) const
{
    const auto vec = tensor.vec<int>();

    vector<int> result;
    result.reserve(vec.size() + 2);
    result.push_back(0);
    for(int i = 0; i < vec.size(); ++i) {
        result.push_back(vec(i));
    }
    result.push_back(dimensionSize);
    return result;
}

void IndicesCubeOp::setValues(
    const Rect& rect, int value1, int value2, Tensor& result) const
{
    auto resultCube = result.tensor<int, 3>();
    {
        const Eigen::array<int, 3> offsets = {rect.Top, rect.Left, 0};
        const Eigen::array<int, 3> extents = {rect.Height(), rect.Width(), 1};
        resultCube.slice(offsets, extents).setConstant(value1);
    }
    {
        const Eigen::array<int, 3> offsets = {rect.Top, rect.Left, 1};
        const Eigen::array<int, 3> extents = {rect.Height(), rect.Width(), 1};
        resultCube.slice(offsets, extents).setConstant(value2);
    }
}