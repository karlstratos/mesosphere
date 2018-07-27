// Author: Karl Stratos (me@karlstratos.com)
//
// Code for backpropagation and neural network architectures. See the note at:
//
// http://karlstratos.com/notes/backprop.pdf

#ifndef NEURAL_H_
#define NEURAL_H_

#include <Eigen/Dense>
#include <unordered_set>

#include "dag.h"
#include "util_eigen.h"

namespace neural {

// Abstract class for a variable in a computation graph.
class Variable: public dag::Node {
 public:
  // Upon initialization, the following must be done immediately:
  //   (1) Specify parents.
  //   (2) Initialize gradient to zero with correct shape.
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
  // References to value/gradient associated with the variable.
  //
  // Always use ref_value()/ref_gradient() from outside instead of accessing the
  // protected members value_/gradient_. This is because some variables (like
  // InputColumn) are not associated with value_/gradient_ but rather their
  // blocks. They override these methods to return something else, for instance
  // value_.col(i). Eigen::Ref is used to reference either matrices or blocks.
  virtual Eigen::Ref<Eigen::MatrixXd> ref_value() { return value_; }
  virtual Eigen::Ref<Eigen::MatrixXd> ref_gradient() { return gradient_; }

  double get_value(size_t i, size_t j) { return ref_value()(i, j); }
  double get_gradient(size_t i, size_t j) { return ref_gradient()(i, j); }
  double get_value(size_t i);
  double get_gradient(size_t i);

  // Dimensions are inferred from gradient.
  std::string Shape() { return util_eigen::dimension_string(ref_gradient()); }
  size_t NumRows() { return ref_gradient().rows(); }
  size_t NumColumns() { return ref_gradient().cols(); }

  // Calculates value and pushes the variable to the topological order if given
  // one and has not been appended yet. Returns the computed value.
  Eigen::Ref<Eigen::MatrixXd> Forward(std::vector<std::shared_ptr<Variable>>
                                      *topological_order);
  Eigen::Ref<Eigen::MatrixXd> Forward() { return Forward(nullptr); }

  // Calculates value from parents (assumed to have their values).
  virtual void ComputeValue() = 0;

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

  std::shared_ptr<Variable> Parent(size_t i) {
    return std::static_pointer_cast<Variable>(dag::Node::Parent(i));
  }
  std::shared_ptr<Variable> Child(size_t i) {
    return std::static_pointer_cast<Variable>(dag::Node::Child(i));
  }
  std::shared_ptr<Variable> shared_from_this() {
    return std::static_pointer_cast<Variable>(dag::Node::shared_from_this());
  }

 protected:
  Eigen::MatrixXd value_;
  Eigen::MatrixXd gradient_;
  bool appended_to_topological_order_ = false;
};

// X: Input is a special variable. rather than maintaining its own value and
// gradient, it only keeps the addresses of some external memory blocks whose
// lifetime subsumes its own lifetime. These addresses must never be corrupted.
class Input: public Variable {
 public:
  Input(Eigen::MatrixXd *value_address, Eigen::MatrixXd *gradient_address);
  void ComputeValue() override { }  // No value to compute.
  void PropagateGradient() override { }  // No parents to propagate to.

  Eigen::Ref<Eigen::MatrixXd> ref_value() override { return *value_address_; }
  Eigen::Ref<Eigen::MatrixXd> ref_gradient() override {
    return *gradient_address_;
  }
 protected:
  Eigen::MatrixXd *value_address_;
  Eigen::MatrixXd *gradient_address_;
};

// X.col(i): InputColumn is Input at a certain column.
class InputColumn: public Input {
 public:
  InputColumn(Eigen::MatrixXd *value_address, Eigen::MatrixXd *gradient_address,
              size_t column_index);
  Eigen::Ref<Eigen::MatrixXd> ref_value() override {
    return value_address_->col(column_index_);  // Reference value column
  }
  Eigen::Ref<Eigen::MatrixXd> ref_gradient() override {
    return gradient_address_->col(column_index_);  // Reference gradient column
  }

 protected:
  size_t column_index_;
};

// X + Y: If X is a non-vector and Y is a vector, assume X + [Y ... Y].
class Add: public Variable {
 public:
  void ComputeValue() override;
  void PropagateGradient() override;
  void set_matrix_vector(bool matrix_vector) { matrix_vector_ = matrix_vector; }
  bool matrix_vector() { return matrix_vector_; }
 protected:
  bool matrix_vector_ = false;
};

// X + c: element-wise
class AddScalar: public Variable {
 public:
  void ComputeValue() override {
    value_ = Parent(0)->ref_value().array() + scalar_value_;
  }
  void PropagateGradient() override { Parent(0)->ref_gradient() += gradient_; }
  void set_scalar_value(double scalar_value) { scalar_value_ = scalar_value; }
 protected:
  double scalar_value_;
};

// X - Y: If X is a non-vector and Y is a vector, assume X - [Y ... Y].
class Subtract: public Variable {
 public:
  void ComputeValue() override;
  void PropagateGradient() override;
  void set_matrix_vector(bool matrix_vector) { matrix_vector_ = matrix_vector; }
  bool matrix_vector() { return matrix_vector_; }
 protected:
  bool matrix_vector_ = false;
};

// sum_i X_i
class ReduceSum: public Variable {
 public:
  void ComputeValue() override {
    value_ = Eigen::MatrixXd::Constant(1, 1, Parent(0)->ref_value().sum());
  }
  void PropagateGradient() override {
    Parent(0)->ref_gradient().array() += gradient_(0);
  }
};

// (1/n) sum_i X_i
class ReduceAverage: public Variable {
 public:
  void ComputeValue() override {
    value_ = Eigen::MatrixXd::Constant(1, 1, Parent(0)->ref_value().mean());
  }
  void PropagateGradient() override {
    Parent(0)->ref_gradient().array() +=
        gradient_(0) / Parent(0)->ref_value().size();
  }
};

// X * Y: linear algebraic matrix-matrix multiplication
class  Multiply: public Variable {
 public:
  void ComputeValue() override {
    value_ = Parent(0)->ref_value() * Parent(1)->ref_value();
  }
  void PropagateGradient() override;
};

// X .* Y: element-wise matrix-matrix multiplication
class MultiplyElementwise: public Variable {
 public:
  void ComputeValue() override {
    value_.array() = Parent(0)->ref_value().array() *
                     Parent(1)->ref_value().array();
  }
  void PropagateGradient() override;
};

// X * c: element-wise matrix-scalar multiplication
class MultiplyScalar: public Variable {
 public:
  void ComputeValue() override {
    value_ = scalar_value_ * Parent(0)->ref_value().array();
  }
  void PropagateGradient() override {
    Parent(0)->ref_gradient().array() += scalar_value_ * gradient_.array(); }
  void set_scalar_value(double scalar_value) { scalar_value_ = scalar_value; }
 protected:
  double scalar_value_;
};

// [X; Y]
class ConcatenateVertical: public Variable {
 public:
  void ComputeValue() override;
  void PropagateGradient() override;
};

// [X, Y]
class ConcatenateHorizontal: public Variable {
 public:
  void ComputeValue() override;
  void PropagateGradient() override;
};

// x^T y: column-wise
class Dot: public Variable {
 public:
  void ComputeValue() override {
    // [a1 a2]'[b1 b2] = [a1'b1 a1'b2;
    //                    a2'b1 a2'b2]
    value_ = (Parent(0)->ref_value().transpose() *  // Eigen should optimize.
              Parent(1)->ref_value()).diagonal().transpose();
  }
  void PropagateGradient() override;
};

// -X
class FlipSign: public Variable {
 public:
  void ComputeValue() override { value_ = -Parent(0)->ref_value(); }
  void PropagateGradient() override { Parent(0)->ref_gradient() -= gradient_; }
};

// X^T
class Transpose: public Variable {
 public:
  void ComputeValue() override { value_ = Parent(0)->ref_value().transpose(); }
  void PropagateGradient() override {
    Parent(0)->ref_gradient() += gradient_.transpose();
  }
};

// 1 / (1 + exp(-x)): element-wise
class Logistic: public Variable {
 public:
  void ComputeValue() override {
    value_ = Parent(0)->ref_value().unaryExpr(
        [](double x) { return 1 / (1 + exp(-x));});
  }
  void PropagateGradient() override {
    Parent(0)->ref_gradient() += gradient_.cwiseProduct(
        value_.unaryExpr([](double x) { return x * (1 - x); }));
  }
};

// tanh(x): element-wise
class Tanh: public Variable {
 public:
  void ComputeValue() override {
    value_ = Parent(0)->ref_value().array().tanh();
  }
  void PropagateGradient() override {
    Parent(0)->ref_gradient() += gradient_.cwiseProduct(
        value_.unaryExpr([](double x) { return 1 - pow(x, 2); }));
  }
};

// relu(x): element-wise
class ReLU: public Variable {
 public:
  void ComputeValue() override {
    value_ = Parent(0)->ref_value().unaryExpr(
        [](double x) { return std::max(0.0, x); });
  }
  void PropagateGradient() override {
    Parent(0)->ref_gradient() += gradient_.cwiseProduct(
        value_.unaryExpr([](double x) { return static_cast<double>(x > 0); }));
  }
};

// softmax(x): column-wise
class Softmax: public Variable {
 public:
  void ComputeValue() override {
    value_ = util_eigen::softmax(Parent(0)->ref_value());
  }
  void PropagateGradient() override;
};

// x_l: column-wise
class Pick: public Variable {
 public:
  Pick(const std::vector<size_t> &indices) : Variable(), indices_(indices) { }
  void ComputeValue() override;
  void PropagateGradient() override;
 protected:
  std::vector<size_t> indices_;
};

// - log [softmax(x)]_l: column-wise
class PickNegativeLogSoftmax: public Variable {
 public:
  PickNegativeLogSoftmax(const std::vector<size_t> &indices) :
      Variable(), indices_(indices) { }
  void ComputeValue() override;
  void PropagateGradient() override;
 protected:
  std::vector<size_t> indices_;
  Eigen::MatrixXd softmax_cache_;
};

// - log (logistic(x))     if true
// - log (1 - logistic(x)) if false: column-wise
class FlagNegativeLogistic: public Variable {
 public:
  FlagNegativeLogistic(const std::vector<bool> &flags) :
      Variable(), flags_(flags) { }
  void ComputeValue() override;
  void PropagateGradient() override;
 protected:
  std::vector<bool> flags_;
  Eigen::MatrixXd logistic_cache_;
};

// A model is a set of weights which perform certain computations.
// To use backpropagation to optimize these weights, their gradients must be
// maintained properly throughout training. This class provides a convenient
// encapsulation of the training details by maintaining, correctly shaping, and
// initializing gradients; it also maintains active shared pointers to prevent
// them from going out of scope.
//
// Note: weights must be all added before making inputs to avoid corrupting
// addresses.
class Model {
 public:
  // Adds a weight and its gradient to holders, returns its index.
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

  // Adds a temporary weight and its temporary gradient to temporary holders
  // (cleared when the current graph is gone), returns its temporary index.
  size_t AddTemporaryWeight(const std::vector<std::vector<double>> &rows) {
    return AddTemporaryWeight(util_eigen::construct_matrix_from_rows(rows));
  }
  size_t AddTemporaryWeight(const Eigen::MatrixXd &temporary_weight);

  // Creates an Input pointer for a weight, resets its gradient to zero,
  // and includes the weight to the update set unless frozen.
  std::shared_ptr<Input> MakeInput(size_t weight_index);

  // Creates an InputColumn pointer for a certain column of the weight, resets
  // the corresponding column of the gradient to zero, and includes the weight
  // column  to the update column set unless frozen.
  std::shared_ptr<InputColumn> MakeInputColumn(size_t weight_index,
                                               size_t column_index);

  // Creates an Input pointer for a temporary weight, resets its temporary
  // gradient to zero. Do not mix indices between model/temporary weights!
  std::shared_ptr<Input> MakeTemporaryInput(size_t temporary_index);

  // Clear intermediate quantities created in the current computation graph.
  // This must be called at each computation during inference to free memory.
  void ClearComputation();

  size_t NumWeights() { return weights_.size(); }
  size_t NumTemporaryWeights() { return temporary_weights_.size(); }

  // We allow external access to weights/gradients because it's sometimes
  // convenient to  dynamically set their values per data instance.
  Eigen::MatrixXd *weight(size_t weight_index) {
    return &weights_[weight_index];
  }
  Eigen::MatrixXd *gradient(size_t weight_index) {
    return &gradients_[weight_index];
  }
  bool frozen(size_t weight_index) { return frozen_[weight_index]; }
  std::unordered_set<size_t> *update_set() { return &update_set_; }
  std::unordered_map<size_t, std::unordered_set<size_t>> *update_column_set() {
    return &update_column_set_;
  }

 private:
  std::vector<Eigen::MatrixXd> weights_;
  std::vector<Eigen::MatrixXd> gradients_;
  std::vector<bool> frozen_;
  std::unordered_set<size_t> update_set_;
  std::unordered_map<size_t, std::unordered_set<size_t>> update_column_set_;
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

  virtual void UpdateWeight(size_t weight_index) = 0;
  virtual void UpdateWeightColumn(size_t weight_index, size_t column_index) = 0;
  double step_size() { return step_size_; }
  void set_step_size(double step_size) { step_size_ = step_size; }

  size_t num_updates(size_t weight_index) { return num_updates_[weight_index]; }
  size_t num_column_updates(size_t weight_index, size_t column_index) {
    return num_column_updates_[weight_index](column_index);
  }

 protected:
  Model *model_;
  std::vector<size_t> num_updates_;
  std::unordered_map<size_t, Eigen::VectorXi> num_column_updates_;
  double step_size_;
};

// Simple gradient descent.
class SimpleGradientDescent: public Updater {
 public:
  SimpleGradientDescent(Model *model, double step_size)
      : Updater(model) { step_size_ = step_size; }
  void UpdateWeight(size_t weight_index) override {
    *model_->weight(weight_index) -= step_size_ *
                                     (*model_->gradient(weight_index));
  }
  void UpdateWeightColumn(size_t weight_index, size_t column_index) override {
    model_->weight(weight_index)->col(column_index) -=
        step_size_ * model_->gradient(weight_index)->col(column_index);
  }
};

// ADAM: https://arxiv.org/pdf/1412.6980.pdf.
class Adam: public Updater {
 public:
  Adam(Model *model, double step_size);
  Adam(Model *model, double step_size, double b1, double b2, double ep);
  void UpdateWeight(size_t weight_index) override;
  void UpdateWeightColumn(size_t weight_index, size_t column_index) override;

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
      size_t num_state_types, Model *model_address);
  virtual ~RNN() { }

  // Computes a sequence of state stacks (each state consisting of different
  // types) for a sequence of observations. The output at indices [t][l][s] is
  // the state at position t in layer l of type s (default state type s=0).
  std::vector<std::vector<std::vector<std::shared_ptr<Variable>>>> Transduce(
      const std::vector<std::shared_ptr<Variable>> &observation_sequence) {
    return Transduce(observation_sequence, {});
  }
  std::vector<std::vector<std::vector<std::shared_ptr<Variable>>>> Transduce(
      const std::vector<std::shared_ptr<Variable>> &observation_sequence,
      const std::vector<std::vector<std::shared_ptr<Variable>>>
      &initial_state_stack);

  // Computes a new state stack. The output at indices [l][s] is the state in
  // layer l of type s.
  std::vector<std::vector<std::shared_ptr<Variable>>> ComputeNewStateStack(
      const std::shared_ptr<Variable> &observation) {
    return ComputeNewStateStack(observation, {}, true);
  }
  std::vector<std::vector<std::shared_ptr<Variable>>> ComputeNewStateStack(
      const std::shared_ptr<Variable> &observation,
      const std::vector<std::vector<std::shared_ptr<Variable>>>
      &previous_state_stack, bool is_beginning=false);

  // Computes a new state for a particular layer l. The output at index [s] is
  // the state in layer l of type s (default state type s=0).
  virtual std::vector<std::shared_ptr<Variable>> ComputeNewState(
      const std::shared_ptr<Variable> &observation,
      const std::vector<std::shared_ptr<Variable>> &previous_state,
      size_t layer) = 0;

  void UseDropout(double dropout_rate, size_t random_seed);
  void StopDropout() { dropout_rate_ = 0.0; }
  void ComputeDropoutWeights();

 protected:
  size_t num_layers_;
  size_t dim_observation_;
  size_t dim_state_;
  size_t num_state_types_;
  size_t batch_size_;
  Model *model_address_;

  std::mt19937 gen_;  // For dropout
  double dropout_rate_ = 0.0;

  // We dynamically set some constant weights per sequence.
  size_t initial_state_index_;
  std::vector<size_t> observation_mask_indices_;
  std::vector<size_t> state_mask_indices_;
};

// Simple RNN.
class SimpleRNN: public RNN {
 public:
  SimpleRNN(size_t num_layers, size_t dim_observation, size_t dim_state,
            Model *model_address);

  std::vector<std::shared_ptr<Variable>> ComputeNewState(
      const std::shared_ptr<Variable> &observation,
      const std::vector<std::shared_ptr<Variable>> &previous_state,
      size_t layer) override;

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

  std::vector<std::shared_ptr<Variable>> ComputeNewState(
      const std::shared_ptr<Variable> &observation,
      const std::vector<std::shared_ptr<Variable>> &previous_state,
      size_t layer) override;

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
};

}  // namespace neural

#endif  // NEURAL_H_
