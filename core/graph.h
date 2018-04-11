// Author: Karl Stratos (me@karlstratos.com)
//
// Graph structures.

#ifndef GRAPH_H_
#define GRAPH_H_

#include <stdlib.h>
#include <string>
#include <vector>

namespace graph {

// A Node object represents a vertex in a directed graph. It serves as a base
// class for graph-structured classes.
//
//   1. It uses a *virtual* destructor so that it can be used polymorphically.
//
//   2. It defines recursive delete functions that will be called from derived
//      classes. Do *not* define their own delete functions in derived classes
//      to avoid complications with Node types.
//
//   3. Properly delete nodes to avoid memory leaks.
//                 (tree)   root->DeleteDescendantsAndSelf();
//        (connected DAG)   sink->DeleteAscendantsAndSelf();
class Node {
 public:
  Node(std::string name) : name_(name) { }
  virtual ~Node() { }

  void DeleteSelfAndAscendants() { DeleteBackward(this); }
  void DeleteSelfAndDescendants() { DeleteForward(this); }

  void AddParent(Node *parent);
  void AddChild(Node *child);

  Node *Parent(size_t i);
  Node *Child(size_t i);

  bool IsRoot() { return parents_.size() == 0; }
  bool IsLeaf() { return children_.size() == 0; }
  size_t NumParents() { return parents_.size(); }
  size_t NumChildren() { return children_.size(); }

  std::string name() { return name_; }
  void set_name(std::string name) { name_ = name; }

 protected:
  std::string name_;  // String name of the node.
  std::vector<Node *> parents_;  // Parent Nodes.
  std::vector<Node *> children_;  // Children Nodes.

 private:
  void DeleteBackward(Node *node);
  void DeleteForward(Node *node);
};

// A TreeNode object represents a vertex in a tree.
class TreeNode: public Node {
 public:
  TreeNode(std::string name) : Node(name) { }
  virtual ~TreeNode() { }

  // Adds a child to the right.
  void AddChildToTheRight(TreeNode *child);

  // Returns the (only) parent of the node if there is one, otherwise nullptr.
  TreeNode *Parent() {
    return (NumParents() > 0) ?
        static_cast<TreeNode *>(Node::Parent(0)) : nullptr;
  }

  // Returns the i-th child node.
  TreeNode *Child(size_t i) { return static_cast<TreeNode *>(Node::Child(i)); }

  // Returns the number of leaves.
  size_t NumLeaves() { return span_end_ - span_begin_ + 1; }

  // Gets the leaves of this node as a sequence of leaf strings.
  void Leaves(std::vector<std::string> *leaf_strings);

  // Returns the string form of this node.
  std::string ToString();

  // Compares the node with the given node.
  bool Compare(TreeNode *node) { return (ToString() == node->ToString()); }

  // Compares the node with the given node string (defined in tree_reader.h).
  bool Compare(std::string node_string);

  // Returns a copy of this node.
  TreeNode *Copy();

  // Sets the span of the node.
  void SetSpan(int span_begin, int span_end);

  size_t child_index() { return child_index_; }
  size_t span_begin() { return span_begin_; }
  size_t span_end() { return span_end_; }
  size_t min_depth() { return min_depth_; }
  size_t max_depth() { return max_depth_; }
  void set_child_index(int child_index) { child_index_ = child_index; }

 protected:
  // Index of this node in the children vector (-1 if not a child).
  int child_index_ = -1;

  // Positions of the first and last leaves that this node spans (-1 if none).
  int span_begin_ = -1;
  int span_end_ = -1;

  // Min/max depth of the node.
  size_t min_depth_ = 0;
  size_t max_depth_ = 0;
};

// Reads a TreeNode structure from a properly formatted string.
class TreeReader {
 public:
  TreeReader() { }
  TreeReader(char open_char, char close_char) : open_char_(open_char),
                                                close_char_(close_char) { }
  ~TreeReader() { }

  // Creates a tree from the given tree string.
  TreeNode *CreateTreeFromTreeString(const std::string &tree_string);

  // Creates a tree from the given token sequence.
  TreeNode *CreateTreeFromTokenSequence(const std::vector<std::string> &toks);

  // Tokenizes the given tree string: "(A (BB	b2))" -> "(", "A", "(", "BB",
  // "b2", ")", ")".
  std::vector<std::string> TokenizeTreeString(const std::string &tree_string);

  void set_open_char(char open_char) { open_char_ = open_char; }
  void set_close_char(char close_char) { close_char_ = close_char; }

 private:
  // Special characters indicating the opening/closing of a subtree.
  char open_char_ = '(';
  char close_char_ = ')';
};

}  // namespace graph

#endif  // GRAPH_H_
