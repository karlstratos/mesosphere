// Author: Karl Stratos (me@karlstratos.com)

#include "neural.h"

namespace neural {

Feedforward::Feedforward(size_t dim_output, size_t dim_input,
                         std::string function,
                         std::vector<autodiff::Input *> *inputs)
    : function_(function) {
  W_value_ = util_eigen::initialize(dim_output, dim_input, "xavier");
  b_value_ = util_eigen::initialize(dim_output, 1, "xavier");
  W_ = new autodiff::Input("W" + std::to_string(inputs->size() + 1), &W_value_);
  inputs->push_back(W_);
  b_ = new autodiff::Input("b" + std::to_string(inputs->size() + 1), &b_value_);
  inputs->push_back(b_);
}

autodiff::Variable *Feedforward::Transform(autodiff::Variable *X) {
  autodiff::Multiply *WX = new autodiff::Multiply(W_, X);
  autodiff::Add *WX_b = new autodiff::Add(WX, b_);
  autodiff::Variable *H;
  if (function_ == "tanh") {
    H =  new autodiff::Tanh(WX_b);
  } else if (function_ == "logistic") {
    H =  new autodiff::Logistic(WX_b);
  } else if (function_ == "relu") {
    H =  new autodiff::ReLU(WX_b);
  } else if (function_ == "identity") {
    H =  WX_b;
  } else {
    ASSERT(false, "Unknown function: " << function_);
  }
  return H;
}

autodiff::Variable *average_negative_log_likelihood(
    autodiff::Variable *X, std::vector<size_t> indices) {
  autodiff::PickNegativeLogSoftmax *L
      = new autodiff::PickNegativeLogSoftmax(X, indices);
  autodiff::ReduceAverage *l = new autodiff::ReduceAverage(L);
  return l;
}

}  // namespace neural
