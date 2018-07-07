// Author: Karl Stratos (me@karlstratos.com)
//
// Implementation of backpropagation. See the note at:
//
// http://karlstratos.com/notes/backprop.pdf

#ifndef AUTODIFF_H_
#define AUTODIFF_H_

#include <Eigen/Dense>
#include <unordered_set>

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

// X: Input is a special variable. rather than maintaining its own value and
// gradient, it only keeps the *addresses* of some external memory blocks whose
// lifetime subsumes its own lifetime. These addresses must never be corrupted.
class Input: public Variable {
 public:
  void Forward(std::vector<std::shared_ptr<Variable>> *topological_order)
      override;
  void PropagateGradient() override { }
  Eigen::MatrixXd *value() override { return value_address_; }
  Eigen::MatrixXd *gradient() override { return gradient_address_; }
  void set_value_address(Eigen::MatrixXd *value_address) {
    value_address_ = value_address;
  }
  void set_gradient_address(Eigen::MatrixXd *gradient_address) {
    gradient_address_ = gradient_address;
  }
 protected:
  Eigen::MatrixXd *value_address_;
  Eigen::MatrixXd *gradient_address_;
  bool called_forward_ = false;
};

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

// A Model is a set of weights (aka. parameters). For convenience, it provides
// functionalities to handle gradients (for a single computation graph). Each
// time it creates an Input variable, it ensures that the corresponding gradient
// is initialized to zero internally. If you wish to work with multiple
// computation graphs at the same time, you should explicitly handle gradients
// yourself outside the model.
class Model {
 public:
  // Adds a weight and returns its index.
  size_t AddWeight(size_t num_rows, size_t num_columns,
                   std::string initialization_method, bool frozen=false) {
    return AddWeight(util_eigen::initialize(num_rows, num_columns,
                                            initialization_method), frozen);
  }
  size_t AddWeight(const std::vector<std::vector<double>> &rows,
                   bool frozen=false) {
    return AddWeight(util_eigen::construct_matrix_from_rows(rows), frozen);
  }
  size_t AddWeight(const Eigen::MatrixXd &weight, bool frozen=false);

  // Adds a temporary weight used only for the current computation graph and
  // returns its temporary index: the weight will be cleared after a model
  // update.
  size_t AddTemporaryWeight(const std::vector<std::vector<double>> &rows) {
    return AddTemporaryWeight(util_eigen::construct_matrix_from_rows(rows));
  }
  size_t AddTemporaryWeight(const Eigen::MatrixXd &temporary_weight);

  // Creates an Input pointer for a weight, initializes its gradient to zero,
  // and includes the weight to the update list unless frozen.
  std::shared_ptr<Input> MakeInput(size_t i);

  // Creates an Input pointer for a temporary weight, initializes its temporary
  // gradient to zero. Note: do not mix indices between permanent weights and
  // temporary weights.
  std::shared_ptr<Input> MakeTemporaryInput(size_t temporary_index);

  // Clear intermediate quantities created in the current computation graph.
  // This must be called at each computation during inference to free memory.
  void ClearComputation();

  size_t NumWeights() { return weights_.size(); }
  size_t NumTemporaryWeights() { return temporary_weights_.size(); }

  Eigen::MatrixXd *weight(size_t i) { return &weights_[i]; }
  Eigen::MatrixXd *gradient(size_t i) { return &gradients_[i]; }
  bool frozen(size_t i) { return frozen_[i]; }
  std::unordered_set<size_t> *update_set() { return &update_set_; }

 private:
  std::vector<Eigen::MatrixXd> weights_;
  std::vector<Eigen::MatrixXd> gradients_;
  std::vector<bool> frozen_;
  std::unordered_set<size_t> update_set_;
  bool made_input_ = false;

  // Holder for temporary input variables that are not part of the model.
  std::vector<Eigen::MatrixXd> temporary_weights_;
  std::vector<Eigen::MatrixXd> temporary_gradients_;
  bool made_temporary_input_ = false;

  // Holder for active variables in the current computation graph (to prevent
  // them from going out of scope and dying).
  std::vector<std::shared_ptr<Variable>> active_variables_;
};

// Abstract class for different model updating schemes.
class Updater {
 public:
  Updater(Model *model) : model_(model) {
    num_updates_.resize(model->NumWeights(), 0);
  }
  virtual ~Updater() { }

  // Update model weights.
  void UpdateWeights();

  virtual void UpdateWeight(size_t i) = 0;
  double step_size() { return step_size_; }
  void set_step_size(double step_size) { step_size_ = step_size; }

 protected:
  Model *model_;
  std::vector<size_t> num_updates_;
  double step_size_;
};

// Simple gradient descent.
class SimpleGradientDescent: public Updater {
 public:
  SimpleGradientDescent(Model *model, double step_size)
      : Updater(model) { step_size_ = step_size; }
  void UpdateWeight(size_t i) override {
    *model_->weight(i) -= step_size_ * (*model_->gradient(i));
  }
};

// ADAM: https://arxiv.org/pdf/1412.6980.pdf.
class Adam: public Updater {
 public:
  Adam(Model *model, double step_size);
  Adam(Model *model, double step_size, double b1, double b2, double ep);
  void UpdateWeight(size_t i) override;

 protected:
  void InitializeMoments();

  double b1_ = 0.9;    // Refresh rate for first-moment gradient est
  double b2_ = 0.999;  // Refresh rate for second-moment gradient est
  double ep_ = 1e-08;  // Prevents division by zero

  std::vector<Eigen::ArrayXXd> first_moments_;
  std::vector<Eigen::ArrayXXd> second_moments_;
};

// Abstract class for recurrent neural networks (RNNs).
class RNN {
 public:
  RNN(size_t num_layers, size_t dim_observation, size_t dim_state,
      Model *model_address) :
      num_layers_(num_layers), dim_observation_(dim_observation),
      dim_state_(dim_state), model_address_(model_address) { }
  virtual ~RNN() { }

  // Computes state stack sequences (corresponding to different state types) for
  // a sequence of observations. The output at indices [s][t][l] is the state of
  // type s at position t in layer l. We use s=0 for the "default" state type.
  std::vector<std::vector<std::vector<std::shared_ptr<Variable>>>> Transduce(
      const std::vector<std::shared_ptr<Variable>> &observation_sequence) {
    return Transduce(observation_sequence, {});
  }
  std::vector<std::vector<std::vector<std::shared_ptr<Variable>>>> Transduce(
      const std::vector<std::shared_ptr<Variable>> &observation_sequence,
      const std::vector<std::shared_ptr<Variable>> &initial_state_stack);

  // Computes a new state stack for the given position.
  void ComputeNewStateStack(
      const std::vector<std::shared_ptr<Variable>> &observation_sequence,
      const std::vector<std::shared_ptr<Variable>> &initial_state_stack,
      size_t position,
      std::vector<std::vector<std::vector<std::shared_ptr<Variable>>>>
      *state_stack_sequences);

  // Computes a new state for the given position and layer.
  virtual void ComputeNewState(
      const std::vector<std::shared_ptr<Variable>> &observation_sequence,
      const std::vector<std::shared_ptr<Variable>> &initial_state_stack,
      size_t position, size_t layer,
      std::vector<std::vector<std::vector<std::shared_ptr<Variable>>>>
      *state_stack_sequences) = 0;

 protected:
  // Gets previous state given a particular state stack sequence (nullptr if
  // none).
  const std::shared_ptr<Variable> &GetPreviousState(
      size_t position, size_t layer,
      const std::vector<std::shared_ptr<Variable>> &initial_state_stack,
      const std::vector<std::vector<std::shared_ptr<Variable>>>
      &particular_state_stack_sequence);

  size_t num_layers_;
  size_t dim_observation_;
  size_t dim_state_;
  size_t num_state_types_ = 1;
  Model *model_address_;
};

// Simple RNN.
class SimpleRNN: public RNN {
 public:
  SimpleRNN(size_t num_layers, size_t dim_observation, size_t dim_state,
            Model *model_address);

  void ComputeNewState(
      const std::vector<std::shared_ptr<Variable>> &observation_sequence,
      const std::vector<std::shared_ptr<Variable>> &initial_state_stack,
      size_t position, size_t layer,
      std::vector<std::vector<std::vector<std::shared_ptr<Variable>>>>
      *state_stack_sequences) override;

  void SetWeights(const Eigen::MatrixXd &U_weight,
                  const Eigen::MatrixXd &V_weight,
                  const Eigen::MatrixXd &b_weight, size_t layer);

 protected:
  std::vector<size_t> U_indices_;
  std::vector<size_t> V_indices_;
  std::vector<size_t> b_indices_;
};

// Long short-term memory (LSTM).
class LSTM: public RNN {
 public:
  LSTM(size_t num_layers, size_t dim_x, size_t dim_h, Model *model_address);

  void ComputeNewState(
      const std::vector<std::shared_ptr<Variable>> &observation_sequence,
      const std::vector<std::shared_ptr<Variable>> &initial_state_stack,
      size_t position, size_t layer,
      std::vector<std::vector<std::vector<std::shared_ptr<Variable>>>>
      *state_stack_sequences) override;

  void SetWeights(const Eigen::MatrixXd &raw_U_weight,
                  const Eigen::MatrixXd &raw_V_weight,
                  const Eigen::MatrixXd &raw_b_weight,
                  const Eigen::MatrixXd &input_U_weight,
                  const Eigen::MatrixXd &input_V_weight,
                  const Eigen::MatrixXd &input_b_weight,
                  const Eigen::MatrixXd &forget_U_weight,
                  const Eigen::MatrixXd &forget_V_weight,
                  const Eigen::MatrixXd &forget_b_weight,
                  const Eigen::MatrixXd &output_U_weight,
                  const Eigen::MatrixXd &output_V_weight,
                  const Eigen::MatrixXd &output_b_weight,
                  size_t layer);

  void UseDropout(double dropout_rate, size_t random_seed) {
    dropout_rate_ = dropout_rate;
    gen_.seed(random_seed);
  }
  void StopDropout() {
    dropout_rate_ = 0.0;
    NullifyDropoutWeights();
  }
  void NullifyDropoutWeights() {
    for (size_t layer = 0; layer < num_layers_; ++layer) {
      SetDropoutWeights(Eigen::MatrixXd::Zero(0, 0),
                        Eigen::MatrixXd::Zero(0, 0), layer);
    }
  }
  void SetDropoutWeights(const Eigen::MatrixXd &observation_mask_weight,
                         const Eigen::MatrixXd &state_mask_weight,
                         size_t layer);
  void ComputeDropoutWeights(size_t batch_size);

 protected:
  std::vector<size_t> raw_U_indices_;
  std::vector<size_t> raw_V_indices_;
  std::vector<size_t> raw_b_indices_;
  std::vector<size_t> input_U_indices_;
  std::vector<size_t> input_V_indices_;
  std::vector<size_t> input_b_indices_;
  std::vector<size_t> forget_U_indices_;
  std::vector<size_t> forget_V_indices_;
  std::vector<size_t> forget_b_indices_;
  std::vector<size_t> output_U_indices_;
  std::vector<size_t> output_V_indices_;
  std::vector<size_t> output_b_indices_;

  // Dropout masks are dummy parameters that do not participate in gradient
  // updates. Their masking weights will be computed dynamically per sequence.
  std::vector<size_t> observation_mask_indices_;
  std::vector<size_t> state_mask_indices_;
  std::mt19937 gen_;
  double dropout_rate_ = 0.0;
};

}  // namespace autodiff

#endif  // AUTODIFF_H_
