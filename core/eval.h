// Author: Karl Stratos (me@karlstratos.com)
//
// Code for evaluation.

#ifndef EVAL_H_
#define EVAL_H_

#include <string>
#include <unordered_map>
#include <vector>

namespace eval {

// Computes per-item and per-sequence accuracy.
std::pair<double, double> compute_accuracy(
    const std::vector<std::vector<std::string>> &true_sequences,
    const std::vector<std::vector<std::string>> &predicted_sequences);

// Computes per-item and per-sequence accuracy by many-to-one mapping.
std::pair<double, double> compute_many2one_accuracy(
    const std::vector<std::vector<std::string>> &true_sequences,
    const std::vector<std::vector<std::string>> &predicted_sequences,
    std::unordered_map<std::string, std::string> *label_mapping=nullptr);

}  // namespace eval

#endif  // EVAL_H_
