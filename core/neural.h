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

// Aliases for readability
template<class T>
using sp = std::shared_ptr<T>;
template<class T>
using sp_v1 = std::vector<std::shared_ptr<T>>;
template<class T>
using sp_v2 = std::vector<std::vector<std::shared_ptr<T>>>;
template<class T>
using sp_v3 = std::vector<std::vector<std::vector<std::shared_ptr<T>>>>;

// Abstract class for a variable in a computation graph.
class Variable: public dag::Node {
 public:
  // Upon initialization, the following must be done immediately:
  //   (1) Specify parents.
  //   (2) Initialize gradient to zero with correct shape.
  //
  // This is done *after* creating a shared pointer to the variable outside
  // (e.g., instead of doing it internally within the constructor) so that weak
  // pointers to the variable can be created/added to parent nodes.
  Variable() : dag::Node() { }

  //------- binary operators ---------------------------------------------------
  friend sp<Variable> operator+(const sp<Variable> &X, const sp<Variable> &Y);
  friend sp<Variable> operator-(const sp<Variable> &X, const sp<Variable> &Y);
  friend sp<Variable> operator+(const sp<Variable> &X, double scalar_value);
  friend sp<Variable> operator+(double scalar_value, const sp<Variable> &X) {
    return X + scalar_value;
  }
  friend sp<Variable> operator-(const sp<Variable> &X, double scalar_value) {
    return X + (-scalar_value);
  }
  // X * Y: linear algebraic matrix-matrix multiplication
  friend sp<Variable> operator*(const sp<Variable> &X, const sp<Variable> &Y);
  // X % Y: element-wise multiplication
  friend sp<Variable> operator%(const sp<Variable> &X, const sp<Variable> &Y);
  friend sp<Variable> operator*(const sp<Variable> &X, double scalar_value);
  friend sp<Variable> operator*(double scalar_value, const sp<Variable> &X) {
    return X * scalar_value;
  }
  friend sp<Variable> operator/(const sp<Variable> &X, double scalar_value) {
    return X * (1.0 / scalar_value);
  }
  friend sp<Variable> operator&(const sp<Variable> &X, const sp<Variable> &Y) {
    return vcat(std::initializer_list<sp<Variable>>{X, Y});
  }
  friend sp<Variable> operator^(const sp<Variable> &X, const sp<Variable> &Y) {
    return hcat(std::initializer_list<sp<Variable>>{X, Y});
  }
  // dot(X, Y): column-wise dot product
  friend sp<Variable> dot(const sp<Variable> &X, const sp<Variable> &Y);
  // cross_entroy(X, Y): column-wise cross entropy where column = distribution
  friend sp<Variable> cross_entropy(const sp<Variable> &X,
                                    const sp<Variable> &Y, bool base2=false) {
    return (base2) ? -sum_cwise(X % log2(Y)) : -sum_cwise(X % log(Y));
  }

  //------- unary operators ----------------------------------------------------
  friend sp<Variable> operator-(const sp<Variable> &X);
  friend sp<Variable> sum(const sp<Variable> &X);
  friend sp<Variable> sum_cwise(const sp<Variable> &X);
  friend sp<Variable> sum_rwise(const sp<Variable> &X);
  friend sp<Variable> average(const sp<Variable> &X) {
    return sum(X) / X->ref_gradient().size();
  }
  friend sp<Variable> average_cwise(const sp<Variable> &X) {
    return sum_cwise(X) / X->ref_gradient().rows();
  }
  friend sp<Variable> average_rwise(const sp<Variable> &X) {
    return sum_rwise(X) / X->ref_gradient().cols();
  }
  friend sp<Variable> transpose(const sp<Variable> &X);
  // squared_norm(X): column-wise squared norm
  friend sp<Variable> squared_norm(const sp<Variable> &X) { return dot(X, X); }
  friend sp<Variable> logistic(const sp<Variable> &X);
  friend sp<Variable> log(const sp<Variable> &X);
  friend sp<Variable> log2(const sp<Variable> &X) {
    return log(X) / std::log(2.0);
  }
  friend sp<Variable> tanh(const sp<Variable> &X);
  friend sp<Variable> relu(const sp<Variable> &X);
  friend sp<Variable> softmax(const sp<Variable> &X);
  friend sp<Variable> entropy(const sp<Variable> &X, bool base2=false) {
    return cross_entropy(X, X, base2);
  }

  //------- list operators -----------------------------------------------------
  friend sp<Variable> sum(const sp_v1<Variable> &Xs);
  friend sp<Variable> average(const sp_v1<Variable> &Xs) {
    return sum(Xs) / Xs.size();
  }
  friend sp<Variable> vcat(const sp_v1<Variable> &Xs);
  friend sp<Variable> hcat(const sp_v1<Variable> &Xs);

  //------- pick operators -----------------------------------------------------
  friend sp<Variable> block(const sp<Variable> &X,
                            size_t start_row, size_t start_column,
                            size_t num_rows, size_t num_columns);
  friend sp<Variable> rows(const sp<Variable> &X,
                           size_t start_row, size_t num_rows) {
    return block(X, start_row, 0, num_rows, X->NumColumns());
  }
  friend sp<Variable> columns(const sp<Variable> &X,
                              size_t start_column, size_t num_columns) {
    return block(X, 0, start_column, X->NumRows(), num_columns);
  }
  friend sp<Variable> column(const sp<Variable> &X, size_t column_index) {
    return block(X, 0, column_index, X->NumRows(), 1);
  }
  friend sp<Variable> row(const sp<Variable> &X, size_t row_index) {
    return block(X, row_index, 0, 1, X->NumColumns());
  }
  friend sp<Variable> pick(const sp<Variable> &X,
                           const std::vector<size_t> &indices);
  friend sp<Variable> cross_entropy(const sp<Variable> &X,
                                    const std::vector<size_t> &indices);
  friend sp<Variable> binary_cross_entropy(const sp<Variable> &X,
                                           const std::vector<bool> &flags);

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
  Eigen::Ref<Eigen::MatrixXd> Forward(sp_v1<Variable> *topological_order);
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
  void Backward(const sp_v1<Variable> &topological_order);

  // Calls Forward and then Backward, returns the final output value.
  double ForwardBackward();

  sp<Variable> Parent(size_t i) {
    return std::static_pointer_cast<Variable>(dag::Node::Parent(i));
  }
  sp<Variable> Child(size_t i) {
    return std::static_pointer_cast<Variable>(dag::Node::Child(i));
  }
  sp<Variable> shared_from_this() {
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

// X.block(i, j, p, q): block of size (p,q), starting at (i,j)
class Block: public Variable {
 public:
  void ComputeValue() override { }  // Value just a block of parent
  void PropagateGradient() override { }  // Gradients propagated from children

  Eigen::Ref<Eigen::MatrixXd> ref_value() override;
  Eigen::Ref<Eigen::MatrixXd> ref_gradient() override;
  void SetBlock(size_t start_row, size_t start_column, size_t num_rows,
                size_t num_columns);
 protected:
  size_t start_row_;
  size_t start_column_;
  size_t num_rows_;
  size_t num_columns_;
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

// sum_i X_i: column-wise
class ReduceSumColumnWise: public Variable {
 public:
  void ComputeValue() override {
    value_ = Parent(0)->ref_value().colwise().sum();
  }
  void PropagateGradient() override {
    Parent(0)->ref_gradient().rowwise() +=
        static_cast<Eigen::RowVectorXd>(gradient_);
  }
};

// sum_i X_i: row-wise
class ReduceSumRowWise: public Variable {
 public:
  void ComputeValue() override {
    value_ = Parent(0)->ref_value().rowwise().sum();
  }
  void PropagateGradient() override {
    Parent(0)->ref_gradient().colwise() +=
        static_cast<Eigen::VectorXd>(gradient_);
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

// X^{(1)} + ... + X^{(n)}: total sum (avoids creating n-1 nodes)
class Sum: public Variable {
 public:
  void ComputeValue() override;
  void PropagateGradient() override;
};

// [X_1;...;X_n]
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

// log(x): element-wise natural logarithm
class Log: public Variable {
 public:
  void ComputeValue() override {
    value_ = Parent(0)->ref_value().array().log();
  }
  void PropagateGradient() override {
    Parent(0)->ref_gradient() += gradient_.cwiseProduct(
        // BUG FIX (8/15/18): log(x) Jacobian is 1/x, not 1/log(x) = 1/value_!
        Parent(0)->ref_value().unaryExpr([](double x) { return 1.0 / x; }));
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
  size_t AddTemporaryWeight(size_t num_rows, size_t num_columns,
                            std::string initialization_method) {
    return AddTemporaryWeight(util_eigen::initialize(num_rows, num_columns,
                                                     initialization_method));
  }
  size_t AddTemporaryWeight(const std::vector<std::vector<double>> &rows) {
    return AddTemporaryWeight(util_eigen::construct_matrix_from_rows(rows));
  }
  size_t AddTemporaryWeight(const Eigen::MatrixXd &temporary_weight);

  // Creates an Input pointer for a weight, resets its gradient to zero,
  // and includes the weight to the update set unless frozen.
  sp<Input> MakeInput(size_t weight_index);

  // Creates an InputColumn pointer for a certain column of the weight, resets
  // the corresponding column of the gradient to zero, and includes the weight
  // column  to the update column set unless frozen.
  sp<InputColumn> MakeInputColumn(size_t weight_index, size_t column_index);

  // Creates an Input pointer for a temporary weight, resets its temporary
  // gradient to zero. Do not mix indices between model/temporary weights!
  sp<Input> MakeTemporaryInput(size_t temporary_index);

  // Clear intermediate quantities created in the current computation graph.
  // This must be called at each computation during inference to free memory.
  void ClearComputation();

  size_t NumWeights() { return weights_.size(); }
  size_t NumTemporaryWeights() { return temporary_weights_.size(); }

  // We allow external access to weights/gradients because it's sometimes
  // convenient to dynamically set their values per data instance.
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

  // Temporary weights are not part of the model and will be cleared.
  std::vector<Eigen::MatrixXd> temporary_weights_;
  std::vector<Eigen::MatrixXd> temporary_gradients_;
  bool made_temporary_input_ = false;

  // Holders for shared pointers to input nodes used in the current computation
  // graph. They serve two purposes:
  //   1. To prevent the nodes from expiring within the current computation
  //   2. To cache: avoid making multiple shared pointers to the same node
  std::unordered_map<size_t, sp<Input>> active_weights_;
  std::unordered_map<size_t, std::unordered_map<size_t, sp<InputColumn>>>
  active_column_weights_;
  std::unordered_map<size_t, sp<Input>> active_temporary_weights_;
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
  sp_v3<Variable> Transduce(const sp_v1<Variable> &observation_sequence) {
    return Transduce(observation_sequence, {});
  }
  sp_v3<Variable> Transduce(const sp_v1<Variable> &observation_sequence,
                            const sp_v2<Variable> &initial_state_stack);

  // Batch unbatched observation sequences with zero padding.
  sp_v1<Variable> Batch(const sp_v2<Variable> &unbatched_observation_sequences);

  // (Xs_1 ... Xs_N) => (HH_1.back().back()[0] ... HH_N.back().back()[0])
  sp_v1<Variable> EncodeByFinalTop(const sp_v2<Variable>
                                   &unbatched_observation_sequences,
                                   bool reverse=false);

  // Computes a new state stack. The output at indices [l][s] is the state in
  // layer l of type s.
  sp_v2<Variable> ComputeNewStateStack(const sp<Variable> &observation) {
    return ComputeNewStateStack(observation, {}, true);
  }
  sp_v2<Variable> ComputeNewStateStack(const sp<Variable> &observation,
                                       const sp_v2<Variable>
                                       &previous_state_stack,
                                       bool is_beginning=false);

  // Computes a new state for a particular layer l. The output at index [s] is
  // the state in layer l of type s (default state type s=0).
  virtual sp_v1<Variable> ComputeNewState(const sp<Variable> &observation,
                                          const sp_v1<Variable> &previous_state,
                                          size_t layer) = 0;

  void UseDropout(double dropout_rate, size_t random_seed);
  void StopDropout() { dropout_rate_ = 0.0; }
  void InitializeDropoutWeights();

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
  size_t empty_observation_index_;
  size_t initial_state_index_;
  std::vector<size_t> observation_mask_indices_;
  std::vector<size_t> state_mask_indices_;
};

// Simple RNN.
class SimpleRNN: public RNN {
 public:
  SimpleRNN(size_t num_layers, size_t dim_observation, size_t dim_state,
            Model *model_address);

  sp_v1<Variable> ComputeNewState(const sp<Variable> &observation,
                                  const sp_v1<Variable> &previous_state,
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

  sp_v1<Variable> ComputeNewState(const sp<Variable> &observation,
                                  const sp_v1<Variable> &previous_state,
                                  size_t layer) override;

  void SetWeights(const Eigen::MatrixXd &U_weight,
                  const Eigen::MatrixXd &V_weight,
                  const Eigen::MatrixXd &b_weight, size_t layer);

 protected:
  std::vector<size_t> U_indices_;
  std::vector<size_t> V_indices_;
  std::vector<size_t> b_indices_;
};

}  // namespace neural

#endif  // NEURAL_H_
