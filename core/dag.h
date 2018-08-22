// Author: Karl Stratos (me@karlstratos.com)
//
// Directed acyclic graph (DAG). See the note on smart pointers at:
//
// http://karlstratos.com/notes/smart_pointers.pdf

#ifndef DAG_H_
#define DAG_H_

#include <stdlib.h>
#include <string>
#include <vector>

namespace dag {

// A Node represents a vertex in a DAG. It serves as a base class for
// DAG-structured objects like trees and computation graphs.
class Node: public std::enable_shared_from_this<Node> {
 public:
  Node() { }
  Node(std::string name) : name_(name) { }

  // Adds the node owned by a shared pointer (passed by constant reference) as
  // a parent/child node.
  void AddParent(const std::shared_ptr<Node> &parent);
  void AddChild(const std::shared_ptr<Node> &child);

  // Creates and returns a shared pointer to the i-th parent/child node.
  std::shared_ptr<Node> Parent(size_t i);
  std::shared_ptr<Node> Child(size_t i);

  bool IsRoot() { return parents_.size() == 0; }
  bool IsLeaf() { return children_.size() == 0; }
  size_t NumParents() { return parents_.size(); }
  size_t NumChildren() { return children_.size(); }

  std::string name() { return name_; }
  void set_name(std::string name) { name_ = name; }

 protected:
  std::string name_;

  // Parent owns children, child points to parents: DAG life = roots life.
  std::vector<std::weak_ptr<Node>> parents_;
  std::vector<std::shared_ptr<Node>> children_;
};

// A TreeNode object represents a vertex in a tree.
class TreeNode: public Node {
 public:
  TreeNode() : Node() { }
  TreeNode(std::string name) : Node(name) { }

  // Adds a child to the right.
  void AddChildToTheRight(const std::shared_ptr<TreeNode> &child);

  // Returns the (only) parent of the node if there is one, otherwise nullptr.
  std::shared_ptr<TreeNode> Parent() {
    return (NumParents() > 0) ?
        std::static_pointer_cast<TreeNode>(Node::Parent(0)) : nullptr;
  }

  // Returns the i-th child node.
  std::shared_ptr<TreeNode> Child(size_t i) {
    return std::static_pointer_cast<TreeNode>(Node::Child(i));
  }

  // Returns the number of leaves.
  size_t NumLeaves() { return span_end_ - span_begin_ + 1; }

  // Gets the leaves of this node as a sequence of leaf strings.
  std::vector<std::string> Leaves();

  // Returns the string form of this node.
  std::string ToString();

  // Compares the node with the given node.
  bool Compare(const std::shared_ptr<TreeNode> &node) {
    return (ToString() == node->ToString());
  }

  // Compares the node with the given node string (defined in tree_reader.h).
  bool Compare(std::string node_string);

  // Returns a copy of this node.
  std::shared_ptr<TreeNode> Copy();

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
  std::shared_ptr<TreeNode> CreateTreeFromTreeString(const std::string
                                                     &tree_string);

  // Creates a tree from the given token sequence.
  std::shared_ptr<TreeNode> CreateTreeFromTokenSequence(
      const std::vector<std::string> &toks);

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

}  // namespace dag

#endif  // DAG_H_
