#include "gc_binarize.h"

#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "min_cut_finder.h"

using namespace tensorflow;

REGISTER_OP("GcBinarize")
    .Input("to_binarize: float32")
    .Input("lambda: float32")
    .Output("binarized: int32")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        return ::tensorflow::shape_inference::UnchangedShapeWithRank(c, 1);
    });

REGISTER_KERNEL_BUILDER(Name("GcBinarize").Device(DEVICE_CPU), GcBinarizeOp);

//////////////////////////////////////////////////////////////////////////////
// GcBinarizeOp

void GcBinarizeOp::Compute(OpKernelContext* context) 
{
  // Grab the input tensor
  const Tensor& probs = context->input(0);
  const Tensor& lambda = context->input(1);
  OP_REQUIRES(context, TensorShapeUtils::IsVector(probs.shape()),
    errors::InvalidArgument("GcBinarize expects a 1-D vector as a first argument."));
  OP_REQUIRES(context, TensorShapeUtils::IsScalar(lambda.shape()),
    errors::InvalidArgument("GcBinarize expects a scalar as a second argument."));

  MinCutFinder::TCapacity capacities;
  const vector<vector<int>> graph = createGraph(probs, lambda.scalar<float>()(0), capacities);
  const MinCutFinder minCutFinder(graph, capacities);
  const vector<bool> partition = minCutFinder.Find(0, graph.size()-1);

  // Create an output tensor
  Tensor* resultTensor = 0;
  OP_REQUIRES_OK(
      context, context->allocate_output(0, probs.shape(), &resultTensor));
  auto resultVector = resultTensor->vec<int>();
  for (int i = 0; i < resultVector.size(); ++i) {
    if(partition[i+1]) {
      resultVector(i) = 1;
    } else {
      resultVector(i) = 0;
    }
  }
}

vector<vector<int>> GcBinarizeOp::createGraph(
  const Tensor& probs, float lambda, 
  MinCutFinder::TCapacity& capacities) const
{
  const auto probsVector = probs.vec<float>();
  const int numOfNodes = probsVector.size() + 2;
  const int s = 0;
  const int t = numOfNodes-1;

  vector<vector<int>> graph(numOfNodes);

  for(int i = 0; i < probsVector.size(); ++i) {
    const float prob = probsVector(i);

    graph[s].push_back(i+1);
    graph[i+1].push_back(s);
    capacities.insert({{s, i+1}, getCapacity(prob)});
    capacities.insert({{i+1, s}, getCapacity(prob)});

    graph[t].push_back(i+1);
    graph[i+1].push_back(t);
    capacities.insert({{t, i+1}, getCapacity(1-prob)});
    capacities.insert({{i+1, t}, getCapacity(1-prob)});
  }

  for(int i = 1; i < probsVector.size(); ++i) {
    graph[i].push_back(i+1);
    graph[i+1].push_back(i);

    capacities.insert({{i, i+1}, getCapacity(lambda)});
    capacities.insert({{i+1, i}, getCapacity(lambda)});
  }

  return graph;
}

int GcBinarizeOp::getCapacity(float value) const
{
  return static_cast<int>(1024 * value);
}