// Author: Karl Stratos (me@karlstratos.com)
//
// Common neural models.

#ifndef NEURAL_H_
#define NEURAL_H_

#include <Eigen/Dense>

#include "autodiff.h"
#include "util_eigen.h"

namespace neural {

// Feedforward model (column-wise).
class Feedforward {
 public:
  Feedforward(size_t dim_output, size_t dim_input, std::string function,
              std::vector<autodiff::Input *> *inputs);
  autodiff::Variable *Transform(autodiff::Variable *X);

  autodiff::Input *W() { return W_; }
  autodiff::Input *b() { return b_; }
  std::string function() const { return function_; }

 private:
  Eigen::MatrixXd W_value_;
  Eigen::MatrixXd b_value_;
  autodiff::Input *W_ = nullptr;
  autodiff::Input *b_ = nullptr;
  std::string function_ = "tanh";
};

// Computes the scalar: -(1/n) sum_i log(softmax_{l_i}(X_i)
autodiff::Variable *average_negative_log_likelihood(
    autodiff::Variable *X, std::vector<size_t> indices);

}  // namespace neural

#endif  // NEURAL_H_
