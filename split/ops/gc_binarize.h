#include <vector>
#include <unordered_map>
#include <utility>

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"

using namespace tensorflow;
using std::vector;
using std::unordered_map;
using std::pair;

class GcBinarizeOp : public OpKernel {
public:
    explicit GcBinarizeOp(OpKernelConstruction* context) : OpKernel(context) {}

    virtual void Compute(OpKernelContext* context) override;

private:
    vector<vector<int>> createGraph(
        const Tensor& probs, float lambda, 
        unordered_map<pair<int, int>, int>& capacities) const;

    int getCapacity(float value) const;
};