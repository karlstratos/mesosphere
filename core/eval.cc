// Author: Karl Stratos (me@karlstratos.com)

#include "eval.h"

#include "util.h"

namespace eval {

std::pair<double, double> compute_accuracy(
    const std::vector<std::vector<std::string>> &true_sequences,
    const std::vector<std::vector<std::string>> &predicted_sequences) {
  size_t num_items = 0;
  size_t num_items_correct = 0;
  size_t num_sequences_correct = 0;
  for (size_t i = 0; i < true_sequences.size(); ++i) {
    num_items += true_sequences[i].size();
    bool entire_sequence_is_correct = true;
    for (size_t j = 0; j < true_sequences[i].size(); ++j) {
      std::string true_string = true_sequences[i][j];
      std::string predicted_string = predicted_sequences[i][j];
      if (predicted_string == true_string) {
        num_items_correct += 1;
      } else {
        entire_sequence_is_correct = false;
      }
    }
    if (entire_sequence_is_correct) { num_sequences_correct += 1; }
  }
  double item_accuracy =
      static_cast<double>(num_items_correct) / num_items * 100;
  double sequence_accuracy =
      static_cast<double>(num_sequences_correct) / true_sequences.size() * 100;
  return std::make_pair(item_accuracy, sequence_accuracy);
}

std::pair<double, double> compute_many2one_accuracy(
    const std::vector<std::vector<std::string>> &true_sequences,
    const std::vector<std::vector<std::string>> &predicted_sequences,
    std::unordered_map<std::string, std::string> *label_mapping) {
  // Create many-to-one label mapping.
  std::unordered_map<std::string, std::string> label_mapping_tmp;
  auto map = (label_mapping == nullptr) ? &label_mapping_tmp : label_mapping;
  std::unordered_map<std::string, std::unordered_map<std::string, size_t> >
      count_matches;
  for (size_t i = 0; i < true_sequences.size(); ++i) {
    for (size_t j = 0; j < true_sequences[i].size(); ++j) {
      ++count_matches[predicted_sequences[i][j]][true_sequences[i][j]];
    }
  }
  for (const auto &predicted_pair: count_matches) {
    std::vector<std::pair<std::string, size_t> > matches;
    for (const auto &true_pair: predicted_pair.second) {
      matches.emplace_back(true_pair.first, true_pair.second);
    }
    std::sort(matches.begin(), matches.end(),
              util_misc::sort_pairs_second<std::string, size_t,
              std::greater<size_t> >());
    (*map)[predicted_pair.first] = matches[0].first;
  }

  // Use the mapping to match label sets.
  std::vector<std::vector<std::string> > predicted_sequences_mapped(
      predicted_sequences.size());
  for (size_t i = 0; i < predicted_sequences.size(); ++i) {
    predicted_sequences_mapped[i].resize(predicted_sequences[i].size());
    for (size_t j = 0; j < predicted_sequences[i].size(); ++j) {
      predicted_sequences_mapped[i][j] = (*map)[predicted_sequences[i][j]];
    }
  }
  return compute_accuracy(true_sequences, predicted_sequences_mapped);
}

}  // namespace eval
