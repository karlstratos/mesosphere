#include <iostream>
#include <stdlib.h>
#include <string>
#include <vector>

#include <Eigen/Dense>

using namespace std;

class Node {
 public:
  Node(string name) : name_(name) { }
  virtual ~Node() { cout << "Node destructor" << endl; }
  void AddParent(Node *parent) {
    parents_.push_back(parent);
    parent->children_.push_back(this);
    size_t before = parent->pindex_.size();
    parent->pindex_.push_back(before);
  }
  Node *Parent(size_t i) const { return parents_[i]; }
  Node *Child(size_t i) const { return children_[i]; }
  size_t NumParents() const { return parents_.size(); }
  size_t NumChildren() const { return children_.size(); }
  string name() const { return name_; }
  size_t pindex(size_t i) const { return pindex_[i]; }
  void NullifyParent(size_t i) { parents_[i] = nullptr; }
  void DeleteAscendantsAndSelf() { DeleteAscendantsAndThis(this); }
  void DeleteAscendantsAndThis(Node *node) {
    for (int i = node->NumParents() - 1; i >= 0; --i) {
      if (node->Parent(i) != nullptr) {
        DeleteAscendantsAndThis(node->Parent(i));
      }
    }
    cout << "Nullifying " << node->name() << " from its children: ";
    for (size_t i = 0; i < node->NumChildren(); ++i) {
      cout <<  node->Child(i)->name() << " ";
      node->Child(i)->NullifyParent(node->pindex(i));
    }
    cout << endl;

    delete node;
  }
  vector<size_t> pindex_;
 protected:
  string name_;
  vector<Node *> parents_;
  vector<Node *> children_;

};

class Variable: public Node {
 public:
  Variable(string name) : Node(name) { }
  ~Variable() { }
  Variable *Parent(size_t i) {
    return static_cast<Variable *>(Node::Parent(i));
  }
  virtual void forward() = 0;
  virtual void backward() = 0;
  size_t NumRows() { return gradient_.rows(); }
  size_t NumColumns() { return gradient_.cols(); }
  virtual Eigen::MatrixXd *value() { return &value_; }
  virtual Eigen::MatrixXd *gradient() { return &gradient_; }
  void make_final() { gradient_ = Eigen::MatrixXd::Ones(1, 1); }
 protected:
  Eigen::MatrixXd value_;
  Eigen::MatrixXd gradient_;
};

struct Input: public Variable {
  Input(string name, Eigen::MatrixXd *input) : Variable(name) {
    input_ = input;
    gradient_ = Eigen::MatrixXd::Zero(input->rows(), input->cols());
  }
  ~Input() { cout << "Deleting " << name_ << endl; }
  Eigen::MatrixXd *value() override { return input_; }
  void forward() override { }
  void backward() override { }
 protected:
  Eigen::MatrixXd *input_;
};

struct Add: public Variable {
  Add(string name, Variable *X, Variable *Y) : Variable(name) {
    AddParent(X);
    AddParent(Y);
    gradient_ = Eigen::MatrixXd::Zero(X->NumRows(), X->NumColumns());

  }
  ~Add() { cout << "Deleting " << name_ << endl; }
  void forward() override {
    Parent(0)->forward();
    Parent(1)->forward();
    value_ = *Parent(0)->value() + *Parent(1)->value();
  }
  void backward() override {
    cout << name_ << " has parents " << Parent(0)->name() << " and "
         << Parent(1)->name() << ", adding "  << gradient_
         << " to each" << endl;
    *Parent(0)->gradient() += gradient_;  // dA = dC
    *Parent(1)->gradient() += gradient_;  // dB = dC
    cout << "recursively calling backward from " << name_
         << " on its first parent " << Parent(0)->name() << endl;
    Parent(0)->backward();
    if (Parent(0) != Parent(1)) {
    cout << "recursively calling backward from " << name_
         << " on its second parent " << Parent(1)->name() << endl;
      Parent(1)->backward();
    }
  }
};

int main() {
  Eigen::MatrixXd x_value(1, 1);
  x_value << 1.0;
  Eigen::MatrixXd y_value(1, 1);
  y_value << 2.0;
  Input *x = new Input("x", &x_value);
  Input *y = new Input("y", &y_value);
  Add *z = new Add("z", x, y);

  Add *q = new Add("q", z, x);
  Add *l = new Add("l", q, q);
  l->make_final();
  l->forward();
  l->backward();
  cout << endl;
  cout << "x = " << *x->value() << endl;
  cout << "dx = " << *x->gradient() << endl;
  cout << endl;
  cout << "y = " << *y->value() << endl;
  cout << "dy = " << *y->gradient() << endl;
  cout << endl;
  cout << "z = " << *z->value() << endl;
  cout << "dz = " << *z->gradient() << endl;
  cout << endl;
  cout << "q = " << *q->value() << endl;
  cout << "dq = " << *q->gradient() << endl;
  cout << endl;
  cout << "l = " << *l->value() << endl;
  cout << "dl = " << *l->gradient() << endl;
  cout << endl;

  x_value += 0.1 * *x->gradient();
  y_value += 0.1 * *y->gradient();

  l->DeleteAscendantsAndSelf();

  cout << endl;
  cout << "x_value is updated to " << x_value << endl;
  cout << "y_value is updated to " << y_value << endl;
  cout << endl;


  return 0;
}
