// Author: Karl Stratos (me@karlstratos.com)

#include "graph.h"

#include <algorithm>
#include <stack>

#include "util.h"

namespace graph {

void Node::AddParent(Node *parent) {
  parents_.push_back(parent);
  parent->children_.push_back(this);

  index_as_child_.push_back(parent->children_.size() - 1);
  parent->index_as_parent_.push_back(parents_.size() - 1);
}

void Node::AddChild(Node *child) {
  children_.push_back(child);
  child->parents_.push_back(this);

  index_as_parent_.push_back(child->parents_.size() - 1);
  child->index_as_child_.push_back(children_.size() - 1);
}

Node *Node::Parent(size_t i) {
  ASSERT(i < NumParents(), "Parent index out of bound: " << i << " / "
         << NumParents());
  return parents_[i];
}

Node *Node::Child(size_t i) {
  ASSERT(i < NumChildren(), "Children index out of bound: " << i << " / "
         << NumChildren());
  return children_[i];
}

void Node::DeleteUniqueSink(Node *unique_sink) {
  for (Node *parent : unique_sink->parents_) {
    // We must check if the parent has already been deleted by its other child:
    //
    //          x - y                * - y           Do not delete x
    //           \ /         =>         /               again from y!
    //            z                    z
    if (parent != nullptr) { DeleteUniqueSink(parent); }
  }
  for (size_t i = 0; i < unique_sink->NumChildren(); ++i) {
    // Disqualify the node as a parent.
    unique_sink->Child(i)->parents_[unique_sink->index_as_parent_[i]] = nullptr;
  }
  delete unique_sink;
}

void Node::DeleteTreeRoot(Node *node) {
  // In a tree, each node is the child of at exactly one node (except the root),
  // so this bottom-up traversal will visit each node exactly once.
  for (Node *child : node->children_) { DeleteTreeRoot(child); }

  delete node;  // Not bothering to disqualify the node as a child parent.
}

void TreeNode::AddChildToTheRight(TreeNode *child) {
  Node::AddChild(child);

  // Adjust the span.
  ASSERT(child->span_begin_ >= 0
         && child->span_end_ >= 0
         && child->span_begin_ <= child->span_end_,
         "Child must have spans define before being added");
  span_begin_ = (span_begin_ >= 0) ? span_begin_ : child->span_begin_;
  span_end_ = child->span_end_;

  // Adjust the depth.
  min_depth_ = (min_depth_ == 0) ? child->min_depth_ + 1 :
               std::min(min_depth_, child->min_depth_ + 1);
  max_depth_ = std::max(max_depth_, child->max_depth_ + 1);
}

void TreeNode::Leaves(std::vector<std::string> *leaf_strings)  {
  leaf_strings->clear();
  std::stack<TreeNode *> dfs_stack;  // Depth-first search (DFS)
  dfs_stack.push(this);
  while (!dfs_stack.empty()) {
    TreeNode *node = dfs_stack.top();
    dfs_stack.pop();
    if (node->IsLeaf()) {
      leaf_strings->push_back(node->name());
    }
    // Putting on the stack right-to-left means popping left-to-right.
    for (int i = node->NumChildren() - 1; i >= 0; --i) {
      dfs_stack.push(node->Child(i));
    }
  }
}

std::string TreeNode::ToString() {
  std::string tree_string = "";
  if (IsLeaf()) {
    tree_string = name_;
  } else {
    std::string children_string = "";
    for (size_t i = 0; i < NumChildren(); ++i) {
      children_string += Child(i)->ToString();
      if (i < NumChildren() - 1) children_string += " ";
    }
    tree_string += "(" + name_ + " " + children_string + ")";
  }
  return tree_string;
}

bool TreeNode::Compare(std::string node_string) {
  TreeReader tree_reader;
  TreeNode *node = tree_reader.CreateTreeFromTreeString(node_string);
  bool is_same = Compare(node);
  node->DeleteThisTreeRoot();
  return is_same;
}

TreeNode *TreeNode::Copy() {
  TreeNode *new_node = new TreeNode(name_);
  new_node->index_as_parent_ = index_as_parent_;
  new_node->index_as_child_ = index_as_child_;
  new_node->span_begin_ = span_begin_;
  new_node->span_end_ = span_end_;
  new_node->min_depth_ = min_depth_;
  new_node->max_depth_ = max_depth_;
  for (size_t i = 0; i < NumChildren(); ++i) {
    new_node->AddChild(Child(i)->Copy());
  }
  return new_node;
}

void TreeNode::SetSpan(int span_begin, int span_end) {
  span_begin_ = span_begin;
  span_end_ = span_end;
}

TreeNode *TreeReader::CreateTreeFromTreeString(const std::string &tree_string) {
  std::vector<std::string> toks = TokenizeTreeString(tree_string);
  TreeNode *tree = CreateTreeFromTokenSequence(toks);
  return tree;
}

TreeNode *TreeReader::CreateTreeFromTokenSequence(const std::vector<std::string>
                                                  &toks) {
  size_t num_left_parentheses = 0;
  size_t num_right_parentheses = 0;
  std::string error_message =
      "Invalid tree string: " + util_string::convert_to_string(toks);

  std::stack<TreeNode *> node_stack;
  size_t leaf_num = 0;  // tracks the position of leaf nodes
  std::string open_string(1, open_char_);
  std::string close_string(1, close_char_);
  for (size_t tok_index = 0; tok_index < toks.size(); ++tok_index) {
    if (toks[tok_index] == open_string) {  // Opening
      ++num_left_parentheses;
      TreeNode *node = new TreeNode("");  // TODO: TreeNode => Node?
      node_stack.push(node);
    } else if (toks[tok_index] == close_string) {  // Closing
      ++num_right_parentheses;
      ASSERT(node_stack.size() > 0, error_message);  // Stack has something.
      if (node_stack.size() <= 1) {
        // We should have reached the end of the tokens.
        ASSERT(tok_index == toks.size() - 1, error_message);

        // Corner case: singleton tree like (a).
        TreeNode *root = node_stack.top();
        if (root->max_depth() == 0) { root->SetSpan(0, 0); }
        break;
      }

      // Otherwise pop node, make it the next child of the top.
      TreeNode *popped_node = node_stack.top();
      node_stack.pop();
      if (popped_node->name().empty()) {
        // If the child is empty, just remove it.
        popped_node->DeleteThisTreeRoot();
        continue;
      } else {
        if (node_stack.top()->name().empty()) {
          // If the parent is empty, skip it.
          TreeNode *parent_node = node_stack.top();
          parent_node->DeleteThisTreeRoot();
          node_stack.pop();
          node_stack.push(popped_node);
        } else {
          // If the parent is non-empty, add the child.
          node_stack.top()->AddChildToTheRight(popped_node);
        }
      }
    } else {
      // We have a symbol.
      if (node_stack.top()->name().empty()) {
        // We must have a non-leaf symbol: ("" => ("NP".
        node_stack.top()->set_name(toks[tok_index]);
      } else {
        // We must have a leaf symbol: ("NP" ("DT" => ("NP" ("DT" "dog".
        // Make this a child of the node on top of the stack.
        TreeNode *leaf = new TreeNode(toks[tok_index]);
        leaf->SetSpan(leaf_num, leaf_num);
        node_stack.top()->AddChildToTheRight(leaf);
        ++leaf_num;
      }
    }
  }
  // There should be a single node on the stack.
  ASSERT(node_stack.size() == 1, error_message);

  // The number of parentheses should match.
  ASSERT(num_left_parentheses == num_right_parentheses, error_message);

  return node_stack.top();
}

std::vector<std::string> TreeReader::TokenizeTreeString(const std::string
                                                        &tree_string) {
  std::vector<std::string> toks;
  std::string tok = "";

  // Are we currently building letters?
  bool building_letters = false;

  for (const char &c : tree_string) {
    if (c == open_char_ || c == close_char_) {  // Delimiter boundary
      if (building_letters) {
        toks.push_back(tok);
        tok = "";
        building_letters = false;
      }
      toks.emplace_back(1, c);
    } else if (c != ' ' && c != '\t') {  // Non-boundary
      building_letters = true;
      tok += c;
    } else { // Empty boundary
      if (building_letters) {
        toks.push_back(tok);
        tok = "";
        building_letters = false;
      }
    }
  }
  return toks;
}

}  // namespace graph
