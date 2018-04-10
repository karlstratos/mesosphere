// Author: Karl Stratos (me@karlstratos.com)
//
// Utility functions. This file must be self-contained.

#ifndef UTIL_H_
#define UTIL_H_

#include <algorithm>
#include <dirent.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <libgen.h>
#include <numeric>
#include <random>
#include <sstream>
#include <stdlib.h>
#include <string>
#include <string.h>
#include <vector>
#include <sys/stat.h>
#include <unordered_map>
#include <vector>

namespace util_string {

// Assert macro that allows adding a message to an assertion upon failure. It
// implictly performs string conversion: ASSERT(x > 0, "Negative x: " << x);
#ifndef NDEBUG
# define ASSERT(condition, message)                                     \
  do {                                                                  \
    if (! (condition)) {                                                \
      std::cerr << "Assertion `" #condition "` failed in " << __FILE__  \
                << " line " << __LINE__ << ": " << message << std::endl; \
      exit(EXIT_FAILURE);                                               \
    }                                                                   \
  } while (false)
#else
#   define ASSERT(condition, message) do { } while (false)
#endif

// Buffers a string to have a certain length.
inline std::string buffer_string(const std::string &given_string, size_t length,
                                 char buffer_char, const std::string &align) {
  std::string buffered_string =
      given_string.substr(0, std::min(given_string.size(), length));
  bool left_turn = true;  // For align = "center".
  std::string buffer(1, buffer_char);
  while (buffered_string.size() < length) {
    if (align == "left" || (align == "center" && left_turn)) {
      buffered_string = buffered_string + buffer;
      left_turn = false;
    } else if (align == "right" || (align == "center" && !left_turn)) {
      buffered_string = buffer + buffered_string;
      left_turn = true;
    } else {
      ASSERT(false, "Unknown alignment method: " << align);
    }
  }
  return buffered_string;
}

// Returns the string form of a printf format string.
inline std::string printf_format(const char *format, ...) {
  char buffer[16384];
  va_list variable_argument_list;
  va_start(variable_argument_list, format);
  vsnprintf(buffer, sizeof(buffer), format, variable_argument_list);
  va_end(variable_argument_list);
  return buffer;
}

// Splits a line by char delimiters.
inline std::vector<std::string> split_by_chars(
    const std::string &line, const std::string &char_delimiters) {
  std::vector<std::string> tokens;
  size_t start = 0;  // Keep track of the current position.
  size_t end = 0;
  std::string token;
  while (end != std::string::npos) {
    // Find the first index a delimiter char occurs.
    end = line.find_first_of(char_delimiters, start);

    // Collect a corresponding portion of the line into a token.
    token = (end == std::string::npos) ? line.substr(start, std::string::npos) :
            line.substr(start, end - start);
    if(token != "") { tokens.push_back(token); }

    // Update the current position.
    start = (end > std::string::npos - 1) ?  std::string::npos : end + 1;
  }
  return tokens;
}

// Splits a line by a string delimiter.
inline std::vector<std::string> split_by_string(
    const std::string &line, const std::string &string_delimiter) {
  std::vector<std::string> tokens;
  size_t start = 0;  // Keep track of the current position.
  size_t end = 0;
  std::string token;
  while (end != std::string::npos) {
    // Find where the string delimiter occurs next.
    end = line.find(string_delimiter, start);

    // Collect a corresponding portion of the line into a token.
    token = (end == std::string::npos) ? line.substr(start, std::string::npos) :
            line.substr(start, end - start);
    if(token != "") { tokens.push_back(token); }

    // Update the current position.
    start = (end > std::string::npos - string_delimiter.size()) ?
            std::string::npos : end + string_delimiter.size();
  }
  return tokens;
}

// Converts seconds to an hour/minute/second string: 6666 => "1h51m6s".
inline std::string convert_seconds_to_string(double num_seconds) {
  size_t num_hours = (int) floor(num_seconds / 3600.0);
  double num_seconds_minus_h = num_seconds - (num_hours * 3600);
  int num_minutes = (int) floor(num_seconds_minus_h / 60.0);
  int num_seconds_minus_hm = num_seconds_minus_h - (num_minutes * 60);
  std::string time_string = std::to_string(num_hours) + "h"
                            + std::to_string(num_minutes) + "m"
                            + std::to_string(num_seconds_minus_hm) + "s";
  return time_string;
}

// Returns an hour/minute/second string of the difftime output.
inline std::string difftime_string(time_t time_now, time_t time_before) {
  double num_seconds = difftime(time_now, time_before);
  return convert_seconds_to_string(num_seconds);
}

// Lowercases a string.
inline std::string lowercase(const std::string &original_string) {
  std::string lowercased_string;
  for (const char &character : original_string) {
    lowercased_string.push_back(tolower(character));
  }
  return lowercased_string;
}

// Converts a value to string up to a certain number of decimal places.
template <typename T>
std::string to_string_with_precision(const T value,
                                     const size_t num_decimal_places = 6) {
  std::ostringstream out;
  out << std::setprecision(num_decimal_places) << value;
  return out.str();
}

// Returns an alphanumeric string of a double, e.g., 1.3503 -> "1p35".
inline std::string convert_to_alphanumeric_string(double value,
                                                  size_t decimal_place) {
  std::string value_string = to_string_with_precision(value, decimal_place);
  for (size_t i = 0; i < value_string.size(); ++i) {
    if (value_string[i] == '.') { value_string[i] = 'p'; }  // Decimal
    if (value_string[i] == '+') { value_string[i] = 'P'; }  // Plus
    if (value_string[i] == '-') { value_string[i] = 'M'; }  // Minus
  }
  return value_string;
}

// Converts a vector to string.
template <typename T>
std::string convert_to_string(const std::vector<T> &sequence) {
  std::string sequence_string;
  for (size_t i = 0; i < sequence.size(); ++i) {
    if (std::is_same<T, std::string>::value) {
      sequence_string += sequence[i];
    } else {
      sequence_string += to_string_with_precision(sequence[i], 2);
    }
    if (i < sequence.size() - 1) { sequence_string += " "; }
  }
  return sequence_string;
}

}  // namespace util_string

namespace util_file {

// Gets the file name from a file path.
inline std::string get_file_name(std::string file_path) {
  return std::string(basename(const_cast<char *>(file_path.c_str())));
}

// Reads the next line from a file into tokens separated by space or tab.
// while (file.good()) {
//     std::vector<std::string> tokens = util::read_line(&file);
//     /* (Do stuff with tokens.) */
// }
inline std::vector<std::string> read_line(std::ifstream *file) {
  std::string line;
  getline(*file, line);
  return util_string::split_by_chars(line, " \t\n");
}

// Returns true if the file exists, false otherwise.
inline bool exists(const std::string &file_path) {
  struct stat buffer;
  return (stat(file_path.c_str(), &buffer) == 0);
}

// Returns the type of the given file path: "file", "dir", or "other".
inline std::string get_file_type(const std::string &file_path) {
  std::string file_type;
  struct stat stat_buffer;
  if (stat(file_path.c_str(), &stat_buffer) == 0) {
    if (stat_buffer.st_mode & S_IFREG) {
      file_type = "file";
    } else if (stat_buffer.st_mode & S_IFDIR) {
      file_type = "dir";
    } else {
      file_type = "other";
    }
  } else {
    ASSERT(false, "Problem with " << file_path);
  }
  return file_type;
}

// Lists files. If given a single file, the list contains the path to that
// file. If given a directory, the list contains the paths to the files
// inside that directory (non-recursively).
inline std::vector<std::string> list_files(const std::string &file_path) {
  std::vector<std::string> list;
  std::string file_type = get_file_type(file_path);
  if (file_type == "dir") {
    DIR *pDIR = opendir(file_path.c_str());
    if (pDIR != NULL) {
      struct dirent *entry = readdir(pDIR);
      while (entry != NULL) {
        if (strcmp(entry->d_name, ".") != 0 &&
            strcmp(entry->d_name, "..") != 0) {
          list.push_back(file_path + "/" + entry->d_name);
        }
        entry = readdir(pDIR);
      }
    }
    closedir(pDIR);
  } else {
    list.push_back(file_path);
  }
  return list;
}

// Returns the number of lines in a file.
inline size_t get_num_lines(const std::string &file_path) {
  size_t num_lines = 0;
  std::string file_type = get_file_type(file_path);
  std::ifstream file(file_path, std::ios::in);
  if (file_type == "file") {
    std::string line;
    while (getline(file, line)) { ++num_lines; }
  }
  return num_lines;
}

// Writes a primitive value to a binary file.
// *WARNING* Do not pass a value stored in a temporary variable (rvalue)!
//     // binary_write_primitive(0, file);  // Bad: undefined behavior
//     size_t zero = 0;
//     binary_write_primitive(zero, file);  // Good
template<typename T>
void binary_write_primitive(const T &lvalue, std::ostream& file){
  file.write(reinterpret_cast<const char *>(&lvalue), sizeof(T));
}

// Reads a primitive value from a binary file.
template<typename T>
void binary_read_primitive(std::istream& file, T *value){
  file.read(reinterpret_cast<char*>(value), sizeof(T));
}

// Writes a primitive unordered_map.
template <typename T1, typename T2>
void binary_write_primitive(const std::unordered_map<T1, T2> &table,
                            const std::string &file_path) {
  std::ofstream file(file_path, std::ios::out | std::ios::binary);
  ASSERT(file.is_open(), "Cannot open file: " << file_path);
  binary_write_primitive(table.size(), file);
  for (const auto &pair : table) {
    binary_write_primitive(pair.first, file);
    binary_write_primitive(pair.second, file);
  }
}

// Reads a primitive unordered_map.
template <typename T1, typename T2>
void binary_read_primitive(const std::string &file_path,
                           std::unordered_map<T1, T2> *table) {
  table->clear();
  std::ifstream file(file_path, std::ios::in | std::ios::binary);
  size_t num_keys;
  binary_read_primitive(file, &num_keys);
  for (size_t i = 0; i < num_keys; ++i) {
    T1 key;
    T2 value;
    binary_read_primitive(file, &key);
    binary_read_primitive(file, &value);
    (*table)[key] = value;
  }
}

// Writes a primitive 2-nested unordered_map.
template <typename T1, typename T2, typename T3>
void binary_write_primitive(
    const std::unordered_map<T1, std::unordered_map<T2, T3>> &table,
    const std::string &file_path) {
  std::ofstream file(file_path, std::ios::out | std::ios::binary);
  ASSERT(file.is_open(), "Cannot open file: " << file_path);
  binary_write_primitive(table.size(), file);
  for (const auto &pair1 : table) {
    binary_write_primitive(pair1.first, file);
    binary_write_primitive(pair1.second.size(), file);
    for (const auto &pair2 : pair1.second) {
      binary_write_primitive(pair2.first, file);
      binary_write_primitive(pair2.second, file);
    }
  }
}

// Reads a primitive 2-nested unordered_map.
template <typename T1, typename T2, typename T3>
void binary_read_primitive(
    const std::string &file_path,
    std::unordered_map<T1, std::unordered_map<T2, T3>> *table) {
  table->clear();
  std::ifstream file(file_path, std::ios::in | std::ios::binary);
  size_t num_first_keys;
  binary_read_primitive(file, &num_first_keys);
  for (size_t i = 0; i < num_first_keys; ++i) {
    T1 first_key;
    size_t num_second_keys;
    binary_read_primitive(file, &first_key);
    binary_read_primitive(file, &num_second_keys);
    for (size_t j = 0; j < num_second_keys; ++j) {
      T2 second_key;
      T3 value;
      binary_read_primitive(file, &second_key);
      binary_read_primitive(file, &value);
      (*table)[first_key][second_key] = value;
    }
  }
}

// Writes a string to a binary file.
inline void binary_write_string(const std::string &value, std::ostream& file) {
  size_t string_length = value.length();
  binary_write_primitive(string_length, file);
  file.write(value.c_str(), string_length);
}

// Reads a string from a binary file.
inline void binary_read_string(std::istream& file, std::string *value) {
  size_t string_length;
  binary_read_primitive(file, &string_length);
  char* buffer = new char[string_length];
  file.read(buffer, string_length);
  value->assign(buffer, string_length);
  delete[] buffer;
}

// Writes a (string, size_t) unordered_map to a binary file.
inline void binary_write(const std::unordered_map<std::string, size_t> &table,
                         const std::string &file_path) {
  std::ofstream file(file_path, std::ios::out | std::ios::binary);
  ASSERT(file.is_open(), "Cannot open file: " << file_path);
  binary_write_primitive(table.size(), file);
  for (const auto &pair : table) {
    binary_write_string(pair.first, file);
    binary_write_primitive(pair.second, file);
  }
}

// Reads a (string, size_t) unordered_map from a binary file.
inline void binary_read(const std::string &file_path,
                        std::unordered_map<std::string, size_t> *table) {
  table->clear();
  std::ifstream file(file_path, std::ios::in | std::ios::binary);
  size_t num_keys;
  binary_read_primitive(file, &num_keys);
  for (size_t i = 0; i < num_keys; ++i) {
    std::string key;
    size_t value;
    binary_read_string(file, &key);
    binary_read_primitive(file, &value);
    (*table)[key] = value;
  }
}

}  // namespace util_file

namespace util_math {

// Returns -inf if a = 0, returns log(a) otherwise (error if a < 0).
inline double log0(double a) {
  if (a > 0.0) {
    return log(a);
  } else if (a == 0.0) {
    return -std::numeric_limits<double>::infinity();
  } else {
    ASSERT(false, "Cannot take log of negative value: " << a);
  }
}

// Given two log values log(a) and log(b), computes log(a + b) without
// exponentiating log(a) and log(b).
inline double sum_logs(double log_a, double log_b) {
  if (log_a < log_b) {
    double temp = log_a;
    log_a = log_b;
    log_b = temp;
  }
  if (log_a <= -std::numeric_limits<double>::infinity()) { return log_a; }

  double negative_difference = log_b - log_a;
  return (negative_difference < -20) ?
      log_a : log_a + log(1.0 + exp(negative_difference));
}

// Computes the average-rank transformation of a sequence of values (e.g., used
// for Spearman's).
template <typename T>
std::vector<double> transform_average_rank(const std::vector<T> &values) {
  std::vector<T> sorted_values(values);
  std::sort(sorted_values.begin(), sorted_values.end());
  std::vector<double> averaged_ranks;
  size_t index = 0;
  while (index < sorted_values.size()) {
    size_t num_same = 1;
    size_t rank_sum = index + 1;
    while (index + 1 < sorted_values.size() &&
           fabs(sorted_values[index + 1] -
                sorted_values[index]) < 1e-15) {
      ++index;
      ++num_same;
      rank_sum += index + 1;
    }

    double averaged_rank = static_cast<double>(rank_sum) / num_same;
    for (size_t j = 0; j < num_same; ++j) {
      // Assign the average rank to all tied elements.
      averaged_ranks.push_back(averaged_rank);
    }
    ++index;
  }

  // Map each value to the corresponding index in averaged_ranks. A value
  // can appear many times but it doesn't matter since it will have the
  // same averaged rank.
  std::unordered_map<double, size_t> value2index;
  for (size_t index = 0; index < sorted_values.size(); ++index) {
    value2index[sorted_values[index]] = index;
  }
  std::vector<double> transformed_values;
  for (double value : values) {
    size_t index = value2index[value];
    transformed_values.push_back(averaged_ranks[index]);
  }
  return transformed_values;
}

// Computes the mean.
template <typename T>
double compute_mean(const std::vector<T> &values) {
  T sum = std::accumulate(values.begin(), values.end(), static_cast<T>(0));
  return static_cast<double>(sum) / values.size();
}

// Computes the covariance.
template <typename T>
double compute_covariance(const std::vector<T> &values1,
                          const std::vector<T> &values2) {
  ASSERT(values1.size() == values2.size() && values1.size() > 1,
         "length1: " << values1.size() << ", length2: " << values2.size());
  double mean1 = compute_mean(values1);
  double mean2 = compute_mean(values2);
  double sum = 0.0;
  for (size_t i = 0; i < values1.size(); ++i) {
    sum += (static_cast<double>(values1.at(i)) - mean1) *
           (static_cast<double>(values2.at(i)) - mean2);
  }
  return sum / (values1.size() - 1);
}

// Computes the variance.
template <typename T>
double compute_variance(const std::vector<T> &values) {
  return compute_covariance(values, values);
}

// Computes the standard deviation of scalars in the given vector.
template <typename T>
double compute_standard_deviation(const std::vector<T> &values) {
  return sqrt(compute_covariance(values, values));
}

// Computes the Pearson correlation coefficient.
template <typename T>
double compute_pearson(const std::vector<T> &values1,
                       const std::vector<T> &values2) {
  double covariance = compute_covariance(values1, values2);
  double standard_deviation1 = compute_standard_deviation(values1);
  double standard_deviation2 = compute_standard_deviation(values2);
  return covariance / standard_deviation1 / standard_deviation2;
}

// Computes the Spearman rank correlation coefficient.
template <typename T>
double compute_spearman(const std::vector<T> &values1,
                        const std::vector<T> &values2) {
  std::vector<double> values1_ranked = transform_average_rank(values1);
  std::vector<double> values2_ranked = transform_average_rank(values2);
  return compute_pearson(values1_ranked, values2_ranked);
}

}  // namespace util_math

namespace util_misc {

// Computes a random permutation of {1...n}, runtime/memory O(n). Needs a random
// seed, for instance:
//   size_t seed = std::chrono::system_clock::now().time_since_epoch().count();
//
// Expected performance: n = 100 million => ~2.5G memory, ~2.5 minutes
inline std::vector<size_t> permute_indices(size_t num_indices, size_t seed) {
  std::vector<size_t> permuted_indices;
  for (size_t i = 0; i < num_indices; ++i) { permuted_indices.push_back(i); }
  std::shuffle(permuted_indices.begin(), permuted_indices.end(),
               std::default_random_engine(seed));
  return permuted_indices;
}

// Template for a struct used to sort a vector of pairs by the second
// values. Use it like this:
//    sort(v.begin(), v.end(), util::sort_pairs_second<int, int>());
//    sort(v.begin(), v.end(),
//         util::sort_pairs_second<int, int, greater<int>>());
template <typename T1, typename T2, typename Predicate = std::less<T2>>
struct sort_pairs_second {
  bool operator()(const std::pair<T1, T2> &left,
                  const std::pair<T1, T2> &right) {
    return Predicate()(left.second, right.second);
  }
};

// Inverts an unordered_map.
template <typename T1, typename T2>
void invert(const std::unordered_map<T1, T2> &table1,
            std::unordered_map<T2, T1> *table2) {
  table2->clear();
  for (const auto &pair : table1) { (*table2)[pair.second] = pair.first; }
}

// Subtracts the median value from all values, guaranteeing the elimination
// of at least half of the elements in the hash table. When counting items
// with only k slots in a stream of N instances, calling this every time all
// slots are filled guarantees that |#(i) - #'(i)| <= 2N/k where #(i) is the
// true count of item i and #'(i) is the approximate count obtained through
// this process.
template <typename T1, typename T2>
void subtract_by_median(std::unordered_map<T1, T2> *table) {
  std::vector<std::pair<T1, T2>> sorted_key_values(table->begin(),
                                                   table->end());
  sort(sorted_key_values.begin(), sorted_key_values.end(),
       sort_pairs_second<T1, T2, std::greater<T2>>());
  T2 median_value = sorted_key_values[(table->size() - 1) / 2].second;

  for (auto iterator = table->begin(); iterator != table->end();) {
    if (iterator->second <= median_value) {
      iterator = table->erase(iterator);
    } else {
      iterator->second -= median_value;
      ++iterator;
    }
  }
}

// Returns the sum of values in an unordered map.
template <typename T1, typename T2>
T2 sum_values(const std::unordered_map<T1, T2> &table) {
  T2 sum = 0.0;
  for (const auto &pair : table) { sum += pair.second; }
  return sum;
}

// Returns the sum of values in a 2-nested unordered map.
template <typename T1, typename T2, typename T3>
T3 sum_values(const std::unordered_map<T1, std::unordered_map<T2, T3>> &table) {
  T3 sum = 0.0;
  for (const auto &pair1 : table) {
    for (const auto &pair2 : pair1.second) { sum += pair2.second; }
  }
  return sum;
}

// Returns true if two unordered maps have the same entries and are close
// in value, else false.
template <typename T1, typename T2>
T2 check_near(const std::unordered_map<T1, T2> &table1,
              const std::unordered_map<T1, T2> &table2) {
  if (table1.size() != table2.size()) { return false; }
  for (const auto &pair1 : table1) {
    T1 key = pair1.first;
    if (table2.find(key) == table2.end()) { return false; }
    if (fabs(table1.at(key) - table2.at(key)) > 1e-10) { return false; }
  }
  return true;
}

// Returns true if two 2-nested unordered maps have the same entries and are
// close in value, else false.
template <typename T1, typename T2, typename T3>
T3 check_near(const std::unordered_map<T1, std::unordered_map<T2, T3>> &table1,
              const std::unordered_map<T1, std::unordered_map<T2, T3>> &table2)
{
  if (table1.size() != table2.size()) { return false; }
  for (const auto &pair1 : table1) {
    T1 key1 = pair1.first;
    if (table2.find(key1) == table2.end()) { return false; }
    if (table1.at(key1).size() != table2.at(key1).size()) {
      return false;
    }
    for (const auto &pair2 : table1.at(key1)) {
      T2 key2 = pair2.first;
      if (table2.at(key1).find(key2) == table2.at(key1).end()) {
        return false;
      }
      if (fabs(table1.at(key1).at(key2) -
               table2.at(key1).at(key2)) > 1e-10) { return false; }
    }
  }
  return true;
}

}  // namespace util_misc

#endif  // UTIL_H_
