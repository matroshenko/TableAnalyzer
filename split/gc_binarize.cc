#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"

using namespace tensorflow;

REGISTER_OP("GcBinarize")
    .Input("to_binarize: float32")
    .Input("lambda: float32")
    .Output("binarized: int32")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        return shape_inference::UnchangedShapeWithRank(c, 1);
    });


class GcBinarizeOp : public OpKernel {
 public:
  explicit GcBinarizeOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& input_tensor = context->input(0);
    OP_REQUIRES(context, TensorShapeUtils::IsVector(input_tensor.shape()),
                errors::InvalidArgument("GcBinarize expects a 1-D vector."));

    auto input = input_tensor.flat<float32>();

    // Create an output tensor
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(
        context, context->allocate_output(0, input_tensor.shape(), &output_tensor));
    auto output_flat = output_tensor->flat<int32>();

    // TODO: Implement binarization.
    const int N = input.size();
    for (int i = 0; i < N; i++) {
      output_flat(i) = 0;
    }
  }
};

REGISTER_KERNEL_BUILDER(Name("GcBinarize").Device(DEVICE_CPU), GcBinarizeOp);