// Author: Karl Stratos (me@karlstratos.com)

#include "gtest/gtest.h"

#include "../graph.h"

void expect_node(graph::TreeNode *node, std::string name, size_t num_parents,
                 graph::TreeNode *parent, size_t num_children, int span_begin,
                 int span_end, size_t min_depth, size_t max_depth) {
  EXPECT_EQ(name, node->name());
  EXPECT_EQ(num_parents, node->NumParents());
  EXPECT_EQ(parent, node->Parent());
  EXPECT_EQ(num_children, node->NumChildren());
  EXPECT_EQ(span_begin, node->span_begin());
  EXPECT_EQ(span_end, node->span_end());
  EXPECT_EQ(min_depth, node->min_depth());
  EXPECT_EQ(max_depth, node->max_depth());
}

void expect_preleaf(graph::TreeNode *node, std::string name, size_t num_parents,
                    graph::TreeNode *parent, int span_index,
                    std::string child_name) {
  expect_node(node, name, num_parents, parent, 1, span_index, span_index, 1, 1);
  expect_node(node->Child(0), child_name, 1, node, 0, span_index, span_index,
              0, 0);
}

class TreeReaderTest : public testing::Test {
 protected:
  graph::TreeReader tree_reader_{'(', ')'};
};

TEST_F(TreeReaderTest, ReadGenericTree) {
  //          TOP
  //           |
  //           AA
  //         / |  \
  //      BBB C*#!  D
  //       |   |
  //      bbb  Q
  //           |
  //         *-1-*
  graph::TreeNode *root = tree_reader_.CreateTreeFromTreeString(
      "(TOP(AA   (BBB	bbb)    (C*#! (Q *-1-*  )) D))");

  // nullptr -> [TOP] -> AA
  expect_node(root, "TOP", 0, nullptr, 1, 0, 2, 2, 4);


  // TOP -> [AA] -> BBB C*#1 D
  graph::TreeNode *child1 = root->Child(0);
  expect_node(child1, "AA", 1, root, 3, 0, 2, 1, 3);

  // AA -> [BBB] -> bbb
  graph::TreeNode *child11 = child1->Child(0);
  expect_preleaf(child11, "BBB", 1, child1, 0, "bbb");

  // AA -> [C*#!] -> Q
  graph::TreeNode *child12 = child1->Child(1);
  expect_node(child12, "C*#!", 1, child1, 1, 1, 1, 2, 2);

  // C*#! -> [Q] -> *-1-*
  graph::TreeNode *child121 = child12->Child(0);
  expect_preleaf(child121, "Q", 1, child12, 1, "*-1-*");

  // AA -> [D]
  graph::TreeNode *child13 = child1->Child(2);
  expect_node(child13, "D", 1, child1, 0, 2, 2, 0, 0);

  root->DeleteSelfAndDescendants();
}

TEST_F(TreeReaderTest, ReadDepth1Tree1Child) {
  //     A
  //     |
  //     a
  graph::TreeNode *root = tree_reader_.CreateTreeFromTreeString("(A a)");

  // nullptr -> [A] -> a
  expect_preleaf(root, "A", 0, nullptr, 0, "a");

  root->DeleteSelfAndDescendants();
}

TEST_F(TreeReaderTest, ReadDepth1TreeManyChildren) {
  //     A
  //    /|\
  //   a b c
  graph::TreeNode *root = tree_reader_.CreateTreeFromTreeString("(A a b c)");

  // nullptr -> [A] -> a b c
  expect_node(root, "A", 0, nullptr, 3, 0, 2, 1, 1);
  expect_node(root->Child(0), "a", 1, root, 0, 0, 0, 0, 0);
  expect_node(root->Child(1), "b", 1, root, 0, 1, 1, 0, 0);
  expect_node(root->Child(2), "c", 1, root, 0, 2, 2, 0, 0);

  root->DeleteSelfAndDescendants();
}

TEST_F(TreeReaderTest, ReadSingletonTree) {
  //     a
  graph::TreeNode *root = tree_reader_.CreateTreeFromTreeString("(a)");

  // nullptr -> [a]
  expect_node(root, "a", 0, nullptr, 0, 0, 0, 0, 0);

  root->DeleteSelfAndDescendants();
}

TEST_F(TreeReaderTest, ReadEmptyTree) {
  graph::TreeNode *root = tree_reader_.CreateTreeFromTreeString("()");
  expect_node(root, "", 0, nullptr, 0, 0, 0, 0, 0);
  root->DeleteSelfAndDescendants();
  root = tree_reader_.CreateTreeFromTreeString("((((()))()()))");
  expect_node(root, "", 0, nullptr, 0, 0, 0, 0, 0);
  root->DeleteSelfAndDescendants();
}

TEST_F(TreeReaderTest, ReadTreeWithExtraBrackets) {
  //           /\
  //          A                    A
  //        / | \                  |
  //         / \          ==       B
  //        B                      |
  //        |                      b
  //        b
  graph::TreeNode *root = tree_reader_.CreateTreeFromTreeString(
      "((A () ((B b) ()) ()) ())");

  // nullptr -> [A] -> B
  expect_node(root, "A", 0, nullptr, 1, 0, 0, 2, 2);

  // A -> [B] -> b
  graph::TreeNode *child1 = root->Child(0);
  expect_preleaf(child1, "B", 1, root, 0, "b");

  root->DeleteSelfAndDescendants();
}

TEST_F(TreeReaderTest, Compare) {
  const std::string &tree1_string = "(TOP (A (B b) (C (D d) (E e) (F f))))";
  const std::string &tree2_string = "(TOP(A(B b)(C(D d)(E e)(F f))))";
  const std::string &tree3_string = "(TOP (A (B b) (C (D d) (E z) (F f))))";
  const std::string &tree4_string = "(TOP (A (Q b) (C (D d) (E e) (F f))))";
  const std::string &tree5_string =
      "(TOP (A (B b) (C (D d) (E e) (F f) (G g))))";
  graph::TreeNode *tree1 = tree_reader_.CreateTreeFromTreeString(tree1_string);
  EXPECT_TRUE(tree1->Compare(tree1_string));
  EXPECT_TRUE(tree1->Compare(tree2_string));
  EXPECT_FALSE(tree1->Compare(tree3_string));
  EXPECT_FALSE(tree1->Compare(tree4_string));
  EXPECT_FALSE(tree1->Compare(tree5_string));
  tree1->DeleteSelfAndDescendants();
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
