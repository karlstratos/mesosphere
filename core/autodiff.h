// Author: Karl Stratos (me@karlstratos.com)
//
// Implementation of backpropagation. See the note at:
//
// http://karlstratos.com/notes/backprop.pdf

#ifndef AUTODIFF_H_
#define AUTODIFF_H_

#include <Eigen/Dense>

#include "dag.h"
#include "util_eigen.h"

namespace autodiff {

// Abstract class for a variable in a computation graph.
class Variable: public dag::Node {
 public:
  // Upon initialization, the following must be done immediately:
  //   (i)  Specify parents.
  //   (ii) Initialize the gradient to zero with a correct shape.
  Variable() : dag::Node() { }
  Variable(std::string name) : dag::Node(name) { }

  //------- binary operators ---------------------------------------------------
  friend std::shared_ptr<Variable> operator+(std::shared_ptr<Variable> X,
                                             std::shared_ptr<Variable> Y);
  friend std::shared_ptr<Variable> operator-(std::shared_ptr<Variable> X,
                                             std::shared_ptr<Variable> Y);
  friend std::shared_ptr<Variable> operator+(std::shared_ptr<Variable> X,
                                             double scalar_value);
  friend std::shared_ptr<Variable> operator+(double scalar_value,
                                             std::shared_ptr<Variable> X) {
    return X + scalar_value;
  }
  friend std::shared_ptr<Variable> operator-(std::shared_ptr<Variable> X,
                                             double scalar_value) {
    return X + (-scalar_value);
  }
  // X * Y: linear algebraic matrix-matrix multiplication
  friend std::shared_ptr<Variable> operator*(std::shared_ptr<Variable> X,
                                             std::shared_ptr<Variable> Y);
  // X % Y: element-wise multiplication
  friend std::shared_ptr<Variable> operator%(std::shared_ptr<Variable> X,
                                             std::shared_ptr<Variable> Y);
  friend std::shared_ptr<Variable> operator*(std::shared_ptr<Variable> X,
                                             double scalar_value);
  friend std::shared_ptr<Variable> operator*(double scalar_value,
                                             std::shared_ptr<Variable> X) {
    return X * scalar_value;
  }
  friend std::shared_ptr<Variable> operator/(std::shared_ptr<Variable> X,
                                             double scalar_value) {
    return X * (1.0 / scalar_value);
  }
  // X & Y: vertical concatenation
  friend std::shared_ptr<Variable> operator&(std::shared_ptr<Variable> X,
                                             std::shared_ptr<Variable> Y);
  // X ^ Y: horizontal concatenation
  friend std::shared_ptr<Variable> operator^(std::shared_ptr<Variable> X,
                                             std::shared_ptr<Variable> Y);
  // dot(X, Y): column-wise dot product
  friend std::shared_ptr<Variable> dot(std::shared_ptr<Variable> X,
                                       std::shared_ptr<Variable> Y);

  //------- unary operators ----------------------------------------------------
  friend std::shared_ptr<Variable> operator-(std::shared_ptr<Variable> X);
  friend std::shared_ptr<Variable> sum(std::shared_ptr<Variable> X);
  friend std::shared_ptr<Variable> average(std::shared_ptr<Variable> X);
  friend std::shared_ptr<Variable> transpose(std::shared_ptr<Variable> X);
  // squared_norm(X): column-wise squared norm
  friend std::shared_ptr<Variable> squared_norm(std::shared_ptr<Variable> X) {
    return dot(X, X);
  }
  friend std::shared_ptr<Variable> logistic(std::shared_ptr<Variable> X);
  friend std::shared_ptr<Variable> tanh(std::shared_ptr<Variable> X);
  friend std::shared_ptr<Variable> relu(std::shared_ptr<Variable> X);
  friend std::shared_ptr<Variable> softmax(std::shared_ptr<Variable> X);

  //------- pick operators -----------------------------------------------------
  friend std::shared_ptr<Variable> pick(std::shared_ptr<Variable> X,
                                        const std::vector<size_t> &indices);
  friend std::shared_ptr<Variable> cross_entropy(
      std::shared_ptr<Variable> X, const std::vector<size_t> &indices);
  friend std::shared_ptr<Variable> binary_cross_entropy(
      std::shared_ptr<Variable> X, const std::vector<bool> &flags);

  //------- class methods ------------------------------------------------------
  // Calculates value from parents, pushes to order (one-time calculation).
  virtual void Forward(std::vector<std::shared_ptr<Variable>>
                       *topological_order) = 0;
  Eigen::MatrixXd Forward();  // Want only values, won't compute gradients.

  // Propagates gradient (assumed complete) to parents by the chain rule.
  virtual void PropagateGradient() = 0;

  // (Meant to be called at a scalar-valued variable at which Forward has
  //  already been called, expects a topological order of variables in the
  //  forward computation.)
  //
  // Calculates gradients of all variables in the graph.
  void Backward(const std::vector<std::shared_ptr<Variable>>
                &topological_order);

  // Calls Forward and then Backward, returns the final output value.
  double ForwardBackward();

  void ResetGradient() { gradient_ = Eigen::MatrixXd::Zero(gradient_.rows(),
                                                           gradient_.cols()); }

  std::string Shape() { return util_eigen::dimension_string(*gradient()); }
  size_t NumRows() { return gradient()->rows(); }
  size_t NumColumns() { return gradient()->cols(); }

  std::shared_ptr<Variable> Parent(size_t i) {
    return std::static_pointer_cast<Variable>(dag::Node::Parent(i));
  }
  std::shared_ptr<Variable> Child(size_t i) {
    return std::static_pointer_cast<Variable>(dag::Node::Child(i));
  }
  std::shared_ptr<Variable> shared_from_this() {
    return std::static_pointer_cast<Variable>(dag::Node::shared_from_this());
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
  void Forward(std::vector<std::shared_ptr<Variable>> *topological_order)
      override;
  void PropagateGradient() override { }
  void set_frozen(bool frozen) { frozen_ = frozen; }
  bool frozen() { return frozen_; }
 protected:
  bool called_forward_ = false;
  bool frozen_ = false;
};

// Make temporary input (frozen = true).
std::shared_ptr<Input> MakeInput(const std::vector<std::vector<double>> &rows);
std::shared_ptr<Input> MakeInput(const Eigen::MatrixXd &value);

// X + Y: If X is a non-vector and Y is a vector, assume X + [Y ... Y].
class Add: public Variable {
 public:
  void Forward(std::vector<std::shared_ptr<Variable>> *topological_order)
      override;
  void PropagateGradient() override;

  void set_matrix_vector(bool matrix_vector) { matrix_vector_ = matrix_vector; }
  bool matrix_vector() { return matrix_vector_; }
 protected:
  bool matrix_vector_ = false;
};

// X + c: element-wise
class AddScalar: public Variable {
 public:
  void Forward(std::vector<std::shared_ptr<Variable>> *topological_order)
      override;
  void PropagateGradient() override { *Parent(0)->gradient() += gradient_; }
  void set_scalar_value(double scalar_value) { scalar_value_ = scalar_value; }
 protected:
  double scalar_value_;
};

// X - Y: If X is a non-vector and Y is a vector, assume X - [Y ... Y].
class Subtract: public Variable {
 public:
  void Forward(std::vector<std::shared_ptr<Variable>> *topological_order)
      override;
  void PropagateGradient() override;

  void set_matrix_vector(bool matrix_vector) { matrix_vector_ = matrix_vector; }
  bool matrix_vector() { return matrix_vector_; }
 protected:
  bool matrix_vector_ = false;
};

// sum_i X_i
class ReduceSum: public Variable {
 public:
  void Forward(std::vector<std::shared_ptr<Variable>> *topological_order)
      override;
  void PropagateGradient() override {
    Parent(0)->gradient()->array() += gradient_(0);
  }
};

// (1/n) sum_i X_i
class ReduceAverage: public Variable {
 public:
  void Forward(std::vector<std::shared_ptr<Variable>> *topological_order)
      override;
  void PropagateGradient() override {
    Parent(0)->gradient()->array() += gradient_(0) / Parent(0)->value()->size();
  }
};

// X * Y: linear algebraic matrix-matrix multiplication
class  Multiply: public Variable {
 public:
  void Forward(std::vector<std::shared_ptr<Variable>> *topological_order)
      override;
  void PropagateGradient() override;
};

// X .* Y: element-wise matrix-matrix multiplication
class MultiplyElementwise: public Variable {
 public:
  void Forward(std::vector<std::shared_ptr<Variable>> *topological_order)
      override;
  void PropagateGradient() override;
};

// X * c: element-wise matrix-scalar multiplication
class MultiplyScalar: public Variable {
 public:
  void Forward(std::vector<std::shared_ptr<Variable>> *topological_order)
      override;
  void PropagateGradient() override {
    Parent(0)->gradient()->array() += scalar_value_ * gradient_.array(); }
  void set_scalar_value(double scalar_value) { scalar_value_ = scalar_value; }
 protected:
  double scalar_value_;
};

// [X; Y]
class ConcatenateVertical: public Variable {
 public:
  void Forward(std::vector<std::shared_ptr<Variable>> *topological_order)
      override;
  void PropagateGradient() override;
};

// [X, Y]
class ConcatenateHorizontal: public Variable {
 public:
  void Forward(std::vector<std::shared_ptr<Variable>> *topological_order)
      override;
  void PropagateGradient() override;
};

// x^T y: column-wise
class Dot: public Variable {
 public:
  void Forward(std::vector<std::shared_ptr<Variable>> *topological_order)
      override;
  void PropagateGradient() override;
};

// -X
class FlipSign: public Variable {
 public:
  void Forward(std::vector<std::shared_ptr<Variable>> *topological_order)
      override;
  void PropagateGradient() override { *Parent(0)->gradient() -= gradient_; }
};

// X^T
class Transpose: public Variable {
 public:
  void Forward(std::vector<std::shared_ptr<Variable>> *topological_order)
      override;
  void PropagateGradient() override {
    *Parent(0)->gradient() += gradient_.transpose();
  }
};

// 1 / (1 + exp(-x)): element-wise
class Logistic: public Variable {
 public:
  void Forward(std::vector<std::shared_ptr<Variable>> *topological_order)
      override;
  void PropagateGradient() override {
    *Parent(0)->gradient() += gradient_.cwiseProduct(
        value()->unaryExpr([](double x) { return x * (1 - x); }));
  }
};

// tanh(x): element-wise
class Tanh: public Variable {
 public:
  void Forward(std::vector<std::shared_ptr<Variable>> *topological_order)
      override;
  void PropagateGradient() override {
    *Parent(0)->gradient() += gradient_.cwiseProduct(
        value()->unaryExpr([](double x) { return 1 - pow(x, 2); }));
  }
};

// relu(x): element-wise
class ReLU: public Variable {
 public:
  void Forward(std::vector<std::shared_ptr<Variable>> *topological_order)
      override;
  void PropagateGradient() override {
    *Parent(0)->gradient() += gradient_.cwiseProduct(
        value()->unaryExpr([](double x) {
            return static_cast<double>(x > 0);
          }));
  }
};

// softmax(x): column-wise
class Softmax: public Variable {
 public:
  void Forward(std::vector<std::shared_ptr<Variable>> *topological_order)
      override;
  void PropagateGradient() override;
};

// x_l: column-wise
class Pick: public Variable {
 public:
  void Forward(std::vector<std::shared_ptr<Variable>> *topological_order)
      override;
  void PropagateGradient() override;
  void set_indices(const std::vector<size_t> &indices) { indices_ = indices; }
 protected:
  std::vector<size_t> indices_;
};

// - log [softmax(x)]_l: column-wise
class PickNegativeLogSoftmax: public Variable {
 public:
  void Forward(std::vector<std::shared_ptr<Variable>> *topological_order)
      override;
  void PropagateGradient() override;
  void set_indices(const std::vector<size_t> &indices) { indices_ = indices; }
 protected:
  std::vector<size_t> indices_;
  Eigen::MatrixXd softmax_cache_;
};

// - log (logistic(x))     if true
// - log (1 - logistic(x)) if false: column-wise
class FlagNegativeLogistic: public Variable {
 public:
  void Forward(std::vector<std::shared_ptr<Variable>> *topological_order)
      override;
  void PropagateGradient() override;
  void set_flags(const std::vector<bool> &flags) { flags_ = flags; }
 protected:
  std::vector<bool> flags_;
  Eigen::MatrixXd logistic_cache_;
};

class InputList {
 public:
  // Creates/initializes an Input.
  std::shared_ptr<Input> Add(std::string name, size_t num_rows,
                             size_t num_columns,
                             std::string initialization_method,
                             bool frozen=false);

  // Creates/fills an Input.
  std::shared_ptr<Input> Add(std::string name,
                             const std::vector<std::vector<double>> &rows,
                             bool frozen=false);
  std::shared_ptr<Input> Add(std::string name,
                             const Eigen::MatrixXd &value,
                             bool frozen=false);
  void Clear();
  size_t Size() { return list_.size(); }
  void ResetGradients() { for (auto X : list_) { X->ResetGradient(); } }

  std::vector<std::shared_ptr<Input>> *list() { return &list_; }
  std::shared_ptr<Input> operator()(size_t i) { return list_[i]; }

 private:
  // DAG illustration                               ___________
  //                                               /           \.
  //          X       Y       Z                  X  __.  Y  ___. Z
  //          .       .       .        =>        .       .       .
  //           \      |      /                    \      |      /
  // list_     [0]   [1]   [2]                    [0]   [1]   [2]
  std::vector<std::shared_ptr<Input>> list_;
};

// Abstract class: different updating schemes for input values.
class Updater {
 public:
  Updater(InputList *inputs) : inputs_(inputs) {
    num_updates_.resize(inputs->Size(), 0);
  }
  virtual ~Updater() { }

  // Update the values and reset the gradients of inputs.
  void UpdateValuesAndResetGradients();

  virtual void UpdateValue(size_t input_index) = 0;
  double step_size() { return step_size_; }
  void set_step_size(double step_size) { step_size_ = step_size; }

 protected:
  InputList *inputs_;
  std::vector<size_t> num_updates_;
  double step_size_;
};

// Simple gradient descent.
class SimpleGradientDescent: public Updater {
 public:
  SimpleGradientDescent(InputList *inputs, double step_size)
      : Updater(inputs) { step_size_ = step_size; }
  void UpdateValue(size_t input_index) override;
};

// ADAM: https://arxiv.org/pdf/1412.6980.pdf.
class Adam: public Updater {
 public:
  Adam(InputList *inputs, double step_size);
  Adam(InputList *inputs, double step_size, double b1, double b2, double ep);
  void UpdateValue(size_t input_index) override;

 protected:
  void InitializeMoments();

  double b1_ = 0.9;    // Refresh rate for first-moment gradient est
  double b2_ = 0.999;  // Refresh rate for second-moment gradient est
  double ep_ = 1e-08;  // Prevents division by zero

  std::vector<Eigen::ArrayXXd> first_moments_;
  std::vector<Eigen::ArrayXXd> second_moments_;
};

}  // namespace autodiff

#endif  // AUTODIFF_H_
