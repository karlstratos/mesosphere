// Author: Karl Stratos (me@karlstratos.com)
//
// Implementation of backpropagation. See the note at:
//
// http://karlstratos.com/notes/backprop.pdf

#ifndef AUTODIFF_H_
#define AUTODIFF_H_

#include <Eigen/Dense>

#include "graph.h"
#include "util_eigen.h"

namespace autodiff {

// Abstract class for a variable in a computation graph.
class Variable: public graph::Node {
 public:
  // Upon initialization, a variable type must
  //   (i)  Specify its parents.
  //   (ii) Initialize the gradient to zero with a correct shape.
  Variable(std::string name) : graph::Node(name) { }

  // Calculates value from parents, pushes to ordering (one-time calculation).
  virtual void Forward(std::vector<Variable *> *topological_ordering) = 0;

  // Propagates gradient (assumed complete) to parents by the chain rule.
  virtual void PropagateGradient() = 0;

  // (Meant to be called at a scalar-valued variable at which Forward has
  //  already been called, expects a topological ordering of variables in the
  //  forward computation.)
  //
  // Calculates gradients of all variables in the graph.
  void Backward(const std::vector<Variable *> &topological_ordering);

  // Calls Forward and then Backward, returns a topological ordering.
  std::vector<Variable *> ForwardBackward();

  std::string Shape() { return util_eigen::dimension_string(*gradient()); }
  size_t NumRows() { return gradient()->rows(); }
  size_t NumColumns() { return gradient()->cols(); }

  Variable *Parent(size_t i) {
    return static_cast<Variable *>(graph::Node::Parent(i));
  }
  Variable *Child(size_t i) {
    return static_cast<Variable *>(graph::Node::Child(i));
  }

  virtual Eigen::MatrixXd *value() { return &value_; }
  virtual Eigen::MatrixXd *gradient() { return &gradient_; }

 protected:
  Eigen::MatrixXd value_;
  Eigen::MatrixXd gradient_;
};

// Input variable that "hooks" onto input values.
class Input: public Variable {
 public:
  Input(std::string name, Eigen::MatrixXd *input);
  void Forward(std::vector<Variable *> *topological_ordering) override;
  void PropagateGradient() override { }  // Has no parent to propagate to.
  void set_input(Eigen::MatrixXd *input) { input_ = input; }

  // Overrides value() so that it points to the external value it's hooked on.
  Eigen::MatrixXd *value() override { return input_; }
 protected:
  Eigen::MatrixXd *input_;  // Pointer to input value.
  bool called_forward_ = false;
};

// Add variable that represents X + Y. If X is a non-vector and Y is a vector,
// this operation is assumed to be X + [Y ... Y].
class Add: public Variable {
 public:
  Add(std::string name, Variable *X, Variable *Y);
  void Forward(std::vector<Variable *> *topological_ordering) override;
  void PropagateGradient() override;
 protected:
  bool matrix_vector_ = false;
};

// Multiply variable that represents X * Y.
struct Multiply: public Variable {
  Multiply(std::string name, Variable *X, Variable *Y);
  void Forward(std::vector<Variable *> *topological_ordering) override;
  void PropagateGradient() override;
};

// Sign-flipping variable that represents -X.
struct FlipSign: public Variable {
  FlipSign(std::string name, Variable *X);
  void Forward(std::vector<Variable *> *topological_ordering) override;
  void PropagateGradient() override { *Parent(0)->gradient() -= gradient_; }
};

// Transpose variable that represents X^T.
struct Transpose: public Variable {
  Transpose(std::string name, Variable *X);
  void Forward(std::vector<Variable *> *topological_ordering) override;
  void PropagateGradient() override {
    *Parent(0)->gradient() += gradient_.transpose();
  }
};

// Logistic variable that represents element-wise 1 / (1 + exp(-X)).
struct Logistic: public Variable {
  Logistic(std::string name, Variable *X);
  void Forward(std::vector<Variable *> *topological_ordering) override;
  void PropagateGradient() override {
    *Parent(0)->gradient() += gradient_.cwiseProduct(
        value()->unaryExpr([](double x) { return x * (1 - x); }));
  }
};

// Tanh variable that represents element-wise tanh(X).
struct Tanh: public Variable {
  Tanh(std::string name, Variable *X);
  void Forward(std::vector<Variable *> *topological_ordering) override;
  void PropagateGradient() override {
    *Parent(0)->gradient() += gradient_.cwiseProduct(
        value()->unaryExpr([](double x) { return 1 - pow(x, 2); }));
  }
};

// Softmax variable that represents softmax(X).
struct Softmax: public Variable {
  Softmax(std::string name, Variable *X);
  void Forward(std::vector<Variable *> *topological_ordering) override;
  void PropagateGradient() override;
};

// Pick variable that represents X_l for some index l.
struct Pick: public Variable {
  Pick(std::string name, Variable *X, size_t index);
  void Forward(std::vector<Variable *> *topological_ordering) override;
  void PropagateGradient() override;
 protected:
  size_t index_;
};

// Pick + negative + log + softmax variable: - log softmax_l(X).
struct PickNegativeLogSoftmax: public Variable {
  PickNegativeLogSoftmax(std::string name, Variable *X, size_t index);
  void Forward(std::vector<Variable *> *topological_ordering) override;
  void PropagateGradient() override;
 protected:
  size_t index_;
  Eigen::MatrixXd cached_value_;
};

}  // namespace autodiff

#endif  // AUTODIFF_H_
