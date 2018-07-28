// Author: Karl Stratos (me@karlstratos.com)

#include "gtest/gtest.h"

#include "../util.h"

TEST(BufferString, Test) {
  std::string foo_left = util_string::buffer_string("foo", 10, '_', "left");
  std::string foo_center = util_string::buffer_string("foo", 10, '_', "center");
  std::string foo_right = util_string::buffer_string("foo", 10, '_', "right");

  EXPECT_EQ("foo_______", foo_left);
  EXPECT_EQ("___foo____", foo_center);
  EXPECT_EQ("_______foo", foo_right);
}

TEST(PrintfFormat, Test) {
  std::string test_string = "TEST_STRING";
  float test_float = 3.14159;
  float test_float_carry = 3.1489;
  size_t test_long = 999999999999999;
  std::string string_string =
      util_string::printf_format("String: %s", test_string.c_str());
  std::string float_string =
      util_string::printf_format("Float: %.2f", test_float);
  std::string float_carry_string =
      util_string::printf_format("Float carry: %.2f", test_float_carry);
  std::string science_string =
      util_string::printf_format("Science: %.2e", test_float);
  std::string long_string =
      util_string::printf_format("Long: %ld", test_long);
  std::string percent_string =
      util_string::printf_format("Percent: 100%%");
  EXPECT_EQ("String: TEST_STRING", string_string);
  EXPECT_EQ("Float: 3.14", float_string);
  EXPECT_EQ("Float carry: 3.15", float_carry_string);
  EXPECT_EQ("Science: 3.14e+00", science_string);
  EXPECT_EQ("Long: 999999999999999", long_string);
  EXPECT_EQ("Percent: 100%", percent_string);
}

TEST(SplitByChars, Test) {
  std::string example = "I have	some\n tabs	and spaces";
  std::vector<std::string> tokens_by_whitespace =
      util_string::split_by_chars(example, " \t\n");
  EXPECT_EQ(6, tokens_by_whitespace.size());
  EXPECT_EQ("I", tokens_by_whitespace[0]);
  EXPECT_EQ("have", tokens_by_whitespace[1]);
  EXPECT_EQ("some", tokens_by_whitespace[2]);
  EXPECT_EQ("tabs", tokens_by_whitespace[3]);
  EXPECT_EQ("and", tokens_by_whitespace[4]);
  EXPECT_EQ("spaces", tokens_by_whitespace[5]);
}

TEST(SplitByString, Test) {
  std::string example = "I have	some\n tabs	and spaces";
  std::vector<std::string> tokens_by_phrase =
      util_string::split_by_string(example, "some\n tabs");
  EXPECT_EQ(2, tokens_by_phrase.size());
  EXPECT_EQ("I have\t", tokens_by_phrase[0]);
  EXPECT_EQ("\tand spaces", tokens_by_phrase[1]);

  std::vector<std::string> tokens_by_space =
      util_string::split_by_string(example, " ");
  EXPECT_EQ(4, tokens_by_space.size());
  EXPECT_EQ("I", tokens_by_space[0]);
  EXPECT_EQ("have	some\n", tokens_by_space[1]);
  EXPECT_EQ("tabs	and", tokens_by_space[2]);
  EXPECT_EQ("spaces", tokens_by_space[3]);
}

TEST(ConvertSecondsToString, Test) {
  EXPECT_EQ("20h7m18s", util_string::convert_seconds_to_string(72438.1));
  EXPECT_EQ("20h7m18s", util_string::convert_seconds_to_string(72438.9));
}

TEST(Lowercase, Test) {
  EXPECT_EQ("ab12345cd@#%! ?ef", util_string::lowercase("AB12345Cd@#%! ?eF"));
}

TEST(ConvertToAlphanumericString, Test) {
  EXPECT_EQ("1p4eP39",
            util_string::convert_to_alphanumeric_string(1.353e+39, 2));
  EXPECT_EQ("M53p1",
            util_string::convert_to_alphanumeric_string(-53.11, 3));
}

TEST(ConvertVectorToString, Test) {
  EXPECT_EQ("a b c",
            util_string::convert_to_string<std::string>({"a", "b", "c"}));
  EXPECT_EQ("1 2 0.54",
            util_string::convert_to_string<double>({1.0, 2.0, 0.538}));
  EXPECT_EQ("0.53", util_string::convert_to_string<float>({0.532}));
  EXPECT_EQ("1 2 3", util_string::convert_to_string<int>({1, 2, 3}));
}

TEST(GetFileName, Test) {
  EXPECT_EQ("a.zip", util_file::get_file_name("../foo/bar/a.zip"));
}

TEST(ReadLine, Test) {
  std::string file_path = "util_file_test_temp";
  std::ofstream file_out(file_path, std::ios::out);
  file_out << "a b	c" << std::endl;
  file_out << std::endl;
  file_out << "		d e f" << std::endl;
  file_out << std::endl;
  file_out.close();
  std::ifstream file_in(file_path, std::ios::in);

  //  "a b\tc"
  std::vector<std::string> tokens = util_file::read_line(&file_in);
  EXPECT_EQ(3, tokens.size());
  EXPECT_EQ("a", tokens[0]);
  EXPECT_EQ("b", tokens[1]);
  EXPECT_EQ("c", tokens[2]);

  //  ""
  tokens = util_file::read_line(&file_in);
  EXPECT_EQ(0, tokens.size());

  //  "\t\td e f"
  tokens = util_file::read_line(&file_in);
  EXPECT_EQ(3, tokens.size());
  EXPECT_EQ("d", tokens[0]);
  EXPECT_EQ("e", tokens[1]);
  EXPECT_EQ("f", tokens[2]);

  //  ""
  tokens = util_file::read_line(&file_in);
  EXPECT_EQ(0, tokens.size());

  //  ""
  tokens = util_file::read_line(&file_in);
  EXPECT_EQ(0, tokens.size());

  file_in.close();
  remove(file_path.c_str());
}

TEST(Exists, Test) {
  std::string file_path = "util_file_test_temp";
  std::ofstream file_out(file_path, std::ios::out);
  file_out.close();
  EXPECT_TRUE(util_file::exists(file_path));
  remove(file_path.c_str());
  EXPECT_FALSE(util_file::exists(file_path));
}

TEST(BinaryWritePrimitive, Test) {
    std::string file_path = "util_file_test_temp";

    std::ofstream file_out(file_path, std::ios::out | std::ios::binary);
    int a = 3;
    util_file::binary_write_primitive(a, file_out);  // lvalue
    float b = 0.39;
    util_file::binary_write_primitive(b, file_out);  // lvalue
    file_out.close();

    std::ifstream file_in(file_path, std::ios::in | std::ios::binary);
    int value1;
    util_file::binary_read_primitive(file_in, &value1);
    EXPECT_EQ(3, value1);
    float value2;
    util_file::binary_read_primitive(file_in, &value2);
    EXPECT_NEAR(0.39, value2, 1e-5);
    file_in.close();

    remove(file_path.c_str());
}

class FileWritingReading : public testing::Test {
protected:
    virtual void SetUp() {
	flat_[0.31] = 7;
	double_nested_[0][-3] = 0.0;
	double_nested_[100][-2] = 1.0 / 3.0;
	flat_string_to_size_t_["The"] = 0;
	flat_string_to_size_t_["elephant"] = 1;
	flat_string_to_size_t_["saw"] = 2;
	flat_string_to_size_t_["."] = 3;
    }
    std::unordered_map<float, size_t> flat_;
    std::unordered_map<size_t, std::unordered_map<int, double>> double_nested_;
    std::unordered_map<std::string, size_t> flat_string_to_size_t_;
    double tol_ = 1e-6;
};

TEST_F(FileWritingReading, FlatUnorderedMapPrimitive) {
    std::string file_path = "util_file_test_temp";
    util_file::binary_write_primitive(flat_, file_path);

    std::unordered_map<float, size_t> table;
    util_file::binary_read_primitive(file_path, &table);
    EXPECT_EQ(1, table.size());
    EXPECT_EQ(7, table[0.31]);
    remove(file_path.c_str());
}

TEST_F(FileWritingReading, DoubleNestedUnorderedMapPrimitive) {
    std::string file_path = "util_file_test_temp";
    util_file::binary_write_primitive(double_nested_, file_path);

    std::unordered_map<size_t, std::unordered_map<int, double>> table;
    util_file::binary_read_primitive(file_path, &table);
    EXPECT_EQ(2, table.size());
    EXPECT_EQ(1, table[0].size());
    EXPECT_NEAR(0.0, table[0][-3], tol_);
    EXPECT_EQ(1, table[100].size());
    EXPECT_NEAR(1.0 / 3.0, table[100][-2], tol_);
    remove(file_path.c_str());
}

TEST_F(FileWritingReading, FlatUnorderedMapStringSizeT) {
    std::string file_path = "util_file_test_temp";
    util_file::binary_write(flat_string_to_size_t_, file_path);

    std::unordered_map<std::string, size_t> table;
    util_file::binary_read(file_path, &table);
    EXPECT_EQ(4, table.size());
    EXPECT_EQ(0, table["The"]);
    EXPECT_EQ(1, table["elephant"]);
    EXPECT_EQ(2, table["saw"]);
    EXPECT_EQ(3, table["."]);
    remove(file_path.c_str());
}

TEST(TransformAverageRank, Test) {
  std::vector<int> sequence =   {3,  -5,   4,   1,   1,  9,  10,  10};
  //           Sorted:          {-5,  1,   1,   3,   4,  9,  10,  10}
  //           Ranks:           <1,   2,   3,   4,   5,  6,   7,   8>
  //           Average ranks:   <1, 2.5, 2.5,   4,   5,  6, 7.5, 7.5>
  //           Unsorted:        <4,   1,   5, 2.5, 2.5,  6, 7.5, 7.5>
  std::vector<double> average_ranks
      = util_math::transform_average_rank(sequence);

  double tol = 1e-10;
  EXPECT_EQ(8, average_ranks.size());
  EXPECT_NEAR(4.0, average_ranks[0], tol);
  EXPECT_NEAR(1.0, average_ranks[1], tol);
  EXPECT_NEAR(5.0, average_ranks[2], tol);
  EXPECT_NEAR(2.5, average_ranks[3], tol);
  EXPECT_NEAR(2.5, average_ranks[4], tol);
  EXPECT_NEAR(6.0, average_ranks[5], tol);
  EXPECT_NEAR(7.5, average_ranks[6], tol);
  EXPECT_NEAR(7.5, average_ranks[7], tol);
}

class Samples : public testing::Test {
 protected:
  virtual void SetUp() {
    values1_ = {56, 75, 45, 71, 61, 64, 58, 80, 76, 61};
    values2_ = {66, 70, 40, 60, 65, 56, 59, 77, 67, 63};
  }
  std::vector<double> values1_;
  std::vector<double> values2_;
};

TEST_F(Samples, ComputeMean) {
  EXPECT_NEAR(64.7, util_math::compute_mean(values1_), 1e-4);
  EXPECT_NEAR(62.3, util_math::compute_mean(values2_), 1e-4);
}

TEST_F(Samples, ComputeCovariance) {
  EXPECT_NEAR(84.8778, util_math::compute_covariance(values1_, values2_), 1e-4);
}

TEST_F(Samples, ComputeVariance) {
  EXPECT_NEAR(116.0111, util_math::compute_variance(values1_), 1e-4);
  EXPECT_NEAR(96.9000, util_math::compute_variance(values2_), 1e-4);
}

TEST_F(Samples, ComputeStandardDeviation) {
  EXPECT_NEAR(10.7708, util_math::compute_standard_deviation(values1_), 1e-4);
  EXPECT_NEAR(9.8438, util_math::compute_standard_deviation(values2_), 1e-4);
}

TEST_F(Samples, ComputePearson) {
  EXPECT_NEAR(0.8005, util_math::compute_pearson(values1_, values2_), 1e-4);
}

TEST_F(Samples, ComputeSpearman) {
  EXPECT_NEAR(0.6687, util_math::compute_spearman(values1_, values2_), 1e-4);
}

TEST(PermuteIndices, Test) {
  size_t num_indices = 100;
  std::vector<size_t> permuted_indices
      = util_misc::permute_indices(num_indices, 42);
  EXPECT_EQ(100, permuted_indices.size());

  bool not_identity = false;
  for (size_t i = 0; i < num_indices; ++i) {
    if (permuted_indices[i] != i) { not_identity = true; }
  }
  EXPECT_TRUE(not_identity);
}

TEST(SortPairsSecond, Test) {
  double tol = 1e-6;
  std::vector<std::pair<std::string, double>> pairs;
  pairs.emplace_back("a", 3.0);
  pairs.emplace_back("b", 0.09);
  pairs.emplace_back("c", 100);

  // Sort in increasing magnitude.
  sort(pairs.begin(), pairs.end(),
       util_misc::sort_pairs_second<std::string, double>());
  EXPECT_EQ("b", pairs[0].first);
  EXPECT_NEAR(0.09, pairs[0].second, tol);
  EXPECT_EQ("a", pairs[1].first);
  EXPECT_NEAR(3.0, pairs[1].second, tol);
  EXPECT_EQ("c", pairs[2].first);
  EXPECT_NEAR(100.0, pairs[2].second, tol);

  // Sort in decreasing magnitude.
  sort(pairs.begin(), pairs.end(),
       util_misc::sort_pairs_second<std::string, double, std::greater<int>>());
  EXPECT_EQ("c", pairs[0].first);
  EXPECT_NEAR(100.0, pairs[0].second, tol);
  EXPECT_EQ("a", pairs[1].first);
  EXPECT_NEAR(3.0, pairs[1].second, tol);
  EXPECT_EQ("b", pairs[2].first);
  EXPECT_NEAR(0.09, pairs[2].second, tol);
}

TEST(SubtractByMedian, Test) {
  std::unordered_map<std::string, size_t> table;
  table["a"] = 100;
  table["b"] = 80;
  table["c"] = 5;
  table["d"] = 3;
  table["e"] = 3;
  table["f"] = 3;
  table["g"] = 1;
  table["h"] = 1;
  table["i"] = 1;
  table["j"] = 1;
  // 100 80 5 3 3 3 1 1 1 1
  //  a  b  c d e f g h i j
  //  0  1  2 3 4 5 6 7 8 9
  //            ^
  //          median
  util_misc::subtract_by_median(&table);

  // Should have a:97, b:77, and c:2 left.
  EXPECT_EQ(3, table.size());
  EXPECT_EQ(97, table["a"]);
  EXPECT_EQ(77, table["b"]);
  EXPECT_EQ(2, table["c"]);
}

TEST(InvertUnorderedMap, Test) {
  std::unordered_map<std::string, size_t> table1;
  table1["a"] = 0;
  table1["b"] = 1;
  auto table2 = util_misc::invert(table1);
  EXPECT_EQ(2, table2.size());
  EXPECT_EQ("a", table2[0]);
  EXPECT_EQ("b", table2[1]);
}

TEST(SumValuesInFlatUnorderedMap, Test) {
  std::unordered_map<std::string, size_t> table;
  table["a"] = 7;
  table["b"] = 3;
  EXPECT_EQ(10, util_misc::sum_values(table));
}

TEST(SumValuesInDoubleNestedUnorderedMap, Test) {
  std::unordered_map<double, std::unordered_map<std::string, int>> table;
  table[0.75]["a"] = -7;
  table[0.75]["b"] = -5;
  table[0.31]["a"] = -3;
  EXPECT_EQ(-15, util_misc::sum_values(table));
}

TEST(CheckNearFlatUnorderedMaps, Test) {
  std::unordered_map<std::string, double> table1;
  std::unordered_map<std::string, double> table2;
  table1["a"] = 7;
  table1["b"] = 7.1;
  table2["a"] = 7;
  table2["b"] = 7.1;
  EXPECT_TRUE(util_misc::check_near(table1, table2));

  table1["c"] = 7.000000001;
  table2["c"] = 7.000000002;
  EXPECT_FALSE(util_misc::check_near(table1, table2));
}

TEST(CheckNearDoubleNestedUnorderedMaps, Test) {
  std::unordered_map<std::string,
                     std::unordered_map<std::string, double>> table1;
  std::unordered_map<std::string,
                     std::unordered_map<std::string, double>> table2;
  table1["a"]["b"] = 7;
  table2["a"]["b"] = 7;
  EXPECT_TRUE(util_misc::check_near(table1, table2));

  table1["a"]["c"] = 7.000000001;
  table2["a"]["c"] = 7.000000002;
  EXPECT_FALSE(util_misc::check_near(table1, table2));
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
