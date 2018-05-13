// Note: Somewhat stale in light of smart pointers.
// Virtual destructor is no longer necessary if we use shared pointers.

#include <iostream>
#include <stdlib.h>
#include <string>
#include <vector>

using namespace std;

class Node {
 public:
  Node(string name) : name_(name) { cout << "Node constructor" << endl; }
  virtual ~Node() { cout << "Node destructor" << endl; }
  void AddParent(Node *parent) {
    parents_.push_back(parent);
    parent->children_.push_back(this);
  }
  void AddChild(Node *child) {
    children_.push_back(child);
    child->parents_.push_back(this);
  }
  Node *Parent(size_t i) const { return parents_[i]; }
  Node *Child(size_t i) const { return children_[i]; }
  size_t NumParents() const { return parents_.size(); }
  size_t NumChildren() const { return children_.size(); }
  string name() const { return name_; }
  void DeleteDescendantsAndSelf() { DeleteDescendantsAndThis(this); }
  void DeleteDescendantsAndThis(Node *node) {
    for (Node *child : node->children_) {
      DeleteDescendantsAndThis(child);
    }
    delete node;
  }
  void DeleteAscendantsAndSelf() { DeleteAscendantsAndThis(this); }
  void DeleteAscendantsAndThis(Node *node) {
    for (Node *parent : node->parents_) {
      DeleteAscendantsAndThis(parent);
    }
    delete node;
  }
 private:
  string name_;
  vector<Node *> parents_;
  vector<Node *> children_;
};

class TreeNode: public Node {
 public:
  TreeNode(string name) : Node(name) { cout << "TreeNode constructor" << endl; }
  virtual ~TreeNode() { cout << "TreeNode destructor" << endl; }
  TreeNode *Child(size_t i) const {
    return static_cast<TreeNode *>(Node::Child(i));
  }
  string ToString() const {
    string tree_string = "";
    if (NumChildren() == 0) {
      tree_string = name();
    } else {
      string children_string = "";
      for (size_t i = 0; i < NumChildren(); ++i) {
        children_string += Child(i)->ToString();
        if (i < NumChildren() - 1) children_string += " ";
      }
      tree_string += "(" + name() + " " + children_string + ")";
    }
    return tree_string;
  }
};

class PCFGNode: public TreeNode {
 public:
  PCFGNode(string name) : TreeNode(name) {
    cout << "PCFGNode constructor" << endl;
  }
  ~PCFGNode() { cout << "PCFGNode destructor" << endl; }
  TreeNode *Child(size_t i) {
    return static_cast<PCFGNode *>(TreeNode::Child(i));
  }
};

class Variable: public Node {
 public:
  Variable(string name) : Node(name) { cout << "Variable constructor" << endl; }
  ~Variable() { cout << "Variable destructor" << endl; }
  virtual void forward() = 0;  // This makes Variable an abstract virtual class.
};

class Input: public Variable {
 public:
  Input(string name) : Variable(name) { cout << "Input constructor" << endl; }
  ~Input() { cout << "Input destructor" << endl; }
  void forward() {
    cout << "Input forward" << endl;
  }
};

class Add: public Variable {
 public:
  Add(string name, Variable *X, Variable *Y) : Variable(name) {
    AddParent(X);
    AddParent(Y);
    cout << "Add constructor" << endl;
  }
  ~Add() { cout << "Add destructor" << endl; }
  void forward() {
    static_cast<Variable *>(Parent(0))->forward();
    static_cast<Variable *>(Parent(1))->forward();
    cout << "Add forward" << endl;
  }
};

int main() {
  cout << "TreeNode * = new TreeNode(...);" << endl;
  TreeNode *x = new TreeNode("x");
  TreeNode *y = new TreeNode("y");
  x->AddChild(y);

  cout << endl;
  cout << x->ToString() << endl;
  cout << endl;

  x->DeleteDescendantsAndSelf();

  cout << endl << endl;
  cout << "PCFGNode * = new PCFGNode(...);" << endl;

  PCFGNode *a = new PCFGNode("a");
  PCFGNode *b = new PCFGNode("b");
  a->AddChild(b);

  cout << endl;
  cout << a->ToString() << endl;
  cout << endl;

  a->DeleteDescendantsAndSelf();

  cout << endl << endl;
  cout << "Input * = new Input(...);" << endl;
  cout << "Add * = new Add(...);" << endl;

  Input *input1 = new Input("input1");
  Input *input2 = new Input("input2");
  Add *add = new Add("add", input1, input2);

  cout << endl;
  add->forward();
  cout << endl;

  add->DeleteAscendantsAndSelf();

  return 0;
}
