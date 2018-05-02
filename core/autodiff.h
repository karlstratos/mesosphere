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
  Eigen::MatrixXd Forward();  // Want only values, won't compute gradients.

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

// X
class Input: public Variable {
 public:
  Input(std::string name, Eigen::MatrixXd *input_value);
  Input(std::string name, Eigen::MatrixXd *input_value,
        std::vector<Input *> *inputs, bool frozen=false);

  void Forward(std::vector<Variable *> *topological_ordering) override;
  void PropagateGradient() override { }
  void set_input_value(Eigen::MatrixXd *input_value) {
    input_value_ = input_value;
  }
  void set_frozen(bool frozen) { frozen_ = frozen; }

  // Overrides value() so that it points to the external value it's hooked on.
  Eigen::MatrixXd *value() override { return input_value_; }
  bool frozen() { return frozen_; }
 protected:
  Eigen::MatrixXd *input_value_;  // Pointer to input value.
  bool called_forward_ = false;
  bool frozen_ = false;
};

// X + Y
class Add: public Variable {
 public:
  // If X is a non-vector and Y is a vector, assume X + [Y ... Y].
  Add(Variable *X, Variable *Y);
  void Forward(std::vector<Variable *> *topological_ordering) override;
  void PropagateGradient() override;
 protected:
  bool matrix_vector_ = false;
};

// sum_i X_i
class ReduceSum: public Variable {
 public:
  ReduceSum(Variable *X);
  void Forward(std::vector<Variable *> *topological_ordering) override;
  void PropagateGradient() override {
    Parent(0)->gradient()->array() += gradient_(0);
  }
};

// (1/n) sum_i X_i
class ReduceAverage: public Variable {
 public:
  ReduceAverage(Variable *X);
  void Forward(std::vector<Variable *> *topological_ordering) override;
  void PropagateGradient() override {
    Parent(0)->gradient()->array() += gradient_(0) / Parent(0)->value()->size();
  }
};

// X * Y
class  Multiply: public Variable {
 public:
  Multiply(Variable *X, Variable *Y);
  void Forward(std::vector<Variable *> *topological_ordering) override;
  void PropagateGradient() override;
};

// dot(X, Y)
class Dot: public Variable {
 public:
  // X and Y can be either row or column.
  Dot(Variable *X, Variable *Y);
  void Forward(std::vector<Variable *> *topological_ordering) override;
  void PropagateGradient() override;
};

// -X
class FlipSign: public Variable {
 public:
  FlipSign(Variable *X);
  void Forward(std::vector<Variable *> *topological_ordering) override;
  void PropagateGradient() override { *Parent(0)->gradient() -= gradient_; }
};

// X^T
class Transpose: public Variable {
 public:
  Transpose(Variable *X);
  void Forward(std::vector<Variable *> *topological_ordering) override;
  void PropagateGradient() override {
    *Parent(0)->gradient() += gradient_.transpose();
  }
};

// 1 / (1 + exp(-x)): element-wise
class Logistic: public Variable {
 public:
  Logistic(Variable *X);
  void Forward(std::vector<Variable *> *topological_ordering) override;
  void PropagateGradient() override {
    *Parent(0)->gradient() += gradient_.cwiseProduct(
        value()->unaryExpr([](double x) { return x * (1 - x); }));
  }
};

// tanh(x): element-wise
class Tanh: public Variable {
 public:
  Tanh(Variable *X);
  void Forward(std::vector<Variable *> *topological_ordering) override;
  void PropagateGradient() override {
    *Parent(0)->gradient() += gradient_.cwiseProduct(
        value()->unaryExpr([](double x) { return 1 - pow(x, 2); }));
  }
};

// softmax(x): column-wise
class Softmax: public Variable {
 public:
  Softmax(Variable *X);
  void Forward(std::vector<Variable *> *topological_ordering) override;
  void PropagateGradient() override;
};

// x_l: column-wise
class Pick: public Variable {
 public:
  Pick(Variable *X, const std::vector<size_t> &indices);
  void Forward(std::vector<Variable *> *topological_ordering) override;
  void PropagateGradient() override;
 protected:
  std::vector<size_t> indices_;
};

// - log [softmax(x)]_l: column-wise
class PickNegativeLogSoftmax: public Variable {
 public:
  PickNegativeLogSoftmax(Variable *X, const std::vector<size_t> &indices);
  void Forward(std::vector<Variable *> *topological_ordering) override;
  void PropagateGradient() override;
 protected:
  std::vector<size_t> indices_;
  Eigen::MatrixXd softmax_cache_;
};

// Abstract class: different updating schemes for input values.
class Updater {
 public:
  Updater(const std::vector<Input *> &inputs)
      : inputs_(inputs) { }
  virtual ~Updater() { }
  virtual void Update() = 0;
  double step_size() { return step_size_; }
  void set_step_size(double step_size) { step_size_ = step_size; }

 protected:
  std::vector<Input *> inputs_;
  double step_size_;
};

// Simple gradient descent.
class SimpleUpdater: public Updater {
 public:
  SimpleUpdater(const std::vector<Input *> &inputs, double step_size = 20.0)
      : Updater(inputs) { step_size_ = step_size; }

  void Update() override {
    for (Input *X : inputs_) {
      if (!X->frozen()) { *X->value() -= step_size_ * *X->gradient(); }
    }
  }
};

}  // namespace autodiff

#endif  // AUTODIFF_H_
