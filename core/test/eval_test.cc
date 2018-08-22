// Author: Karl Stratos (me@karlstratos.com)

#include "gtest/gtest.h"

#include "../eval.h"

TEST(SequenceLabeling, Accuracy) {
  auto pair = eval::compute_accuracy({{"a", "b", "c"}, {"b", "c"}},
                                     {{"a", "c", "c"}, {"b", "c"}});
  EXPECT_NEAR(80.0, pair.first, 1e-15);
  EXPECT_NEAR(50.0, pair.second, 1e-15);
}

TEST(SequenceLabeling, ManyToOneAccuracy) {
  std::unordered_map<std::string, std::string> label_mapping;
  auto pair = eval::compute_many2one_accuracy({{"a", "b", "c"}, {"b", "c"}},
                                              {{"0", "1", "2"}, {"1", "2"}},
                                              &label_mapping);
  EXPECT_NEAR(100.0, pair.first, 1e-15);
  EXPECT_NEAR(100.0, pair.second, 1e-15);

  EXPECT_EQ("a", label_mapping["0"]);
  EXPECT_EQ("b", label_mapping["1"]);
  EXPECT_EQ("c", label_mapping["2"]);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
