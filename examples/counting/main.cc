// Author: Karl Stratos (me@karlstratos.com)

#include <ctime>
#include <limits>
#include <iostream>
#include <stdlib.h>
#include <string>
#include <vector>

#include "../../core/neural.h"

int main (int argc, char* argv[]) {
  size_t random_seed = 42;
  std::string rnn_type = "lstm";
  size_t xdim = 1;
  size_t hdim = 8;
  size_t num_layers = 1;
  size_t N = 5000;
  double step_size = 0.01;
  size_t max_train_length = 100;
  size_t test_length = 200;
  size_t dev_size = 100;
  bool verbose = false;

  // Parse command line arguments.
  bool display_options_and_quit = false;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--seed") {
      random_seed = std::stoi(argv[++i]);
    } else if (arg == "--rnn") {
      rnn_type = argv[++i];
    } else if (arg == "--xdim") {
      xdim = std::stoi(argv[++i]);
    } else if (arg == "--hdim") {
      hdim = std::stoi(argv[++i]);
    } else if (arg == "--N") {
      N = std::stoi(argv[++i]);
    } else if (arg == "--trainlen") {
      max_train_length = std::stoi(argv[++i]);
    } else if (arg == "--testlen") {
      test_length = std::stoi(argv[++i]);
    } else if (arg == "--devsize") {
      dev_size = std::stoi(argv[++i]);
    } else if (arg == "--step") {
      step_size = std::stod(argv[++i]);
    } else if (arg == "--verbose") {
      verbose = true;
    } else if (arg == "--help" || arg == "-h"){
      display_options_and_quit = true;
    } else {
      std::cerr << "Invalid argument \"" << arg << "\": run the command with "
                << "-h or --help to see possible arguments." << std::endl;
      exit(-1);
    }
  }
  if (display_options_and_quit) {
    std::cout << "--seed [" << random_seed << "]:        \t"
              << "random seed" << std::endl;
    std::cout << "--rnn [" << rnn_type << "]:   \t"
              << "choice of RNN" << std::endl;
    std::cout << "--N [" << N << "]:     \t"
              << "number of samples" << std::endl;
    std::cout << "--xdim [" << xdim << "]:        \t"
              << "dimension of observation" << std::endl;
    std::cout << "--hdim [" << hdim << "]:        \t"
              << "dimension of hidden state" << std::endl;
    std::cout << "--trainlen [" << max_train_length << "]:  \t"
              << "max training length (value n in a^n b^n)" << std::endl;
    std::cout << "--testlen [" << max_train_length << "]:   \t"
              << "test length (value n in a^n b^n)" << std::endl;
    std::cout << "--step [" << step_size << "]:        \t"
              << "step size for gradient descent" << std::endl;
    std::cout << "--help, -h:           \t"
              << "show options and quit?" << std::endl;
    exit(0);
  }

  std::srand(random_seed);

  neural::Model model;
  size_t i_a = model.AddWeight(xdim, 1, "unit-variance");
  size_t i_b = model.AddWeight(xdim, 1, "unit-variance");

  std::unique_ptr<neural::RNN> rnn;
  if (rnn_type == "simple") {
    rnn = cc14::make_unique<neural::SimpleRNN>(num_layers, xdim, hdim,
                                               &model);
  } else if (rnn_type == "lstm") {
    rnn = cc14::make_unique<neural::LSTM>(num_layers, xdim, hdim,
                                          &model);
  } else {
    ASSERT(false, "Unknown RNN " << rnn_type);
  }
  size_t i_W = model.AddWeight(1, hdim, "unit-variance");

  neural::Adam gd(&model, step_size);

  std::mt19937 gen(random_seed);
  std::uniform_int_distribution<> dis_length_training(1, max_train_length);
  std::uniform_int_distribution<> dis_negative(0, 1);

  // Create a dev set.
  std::vector<std::pair<int, int>> dev;
  for (size_t i = 0; i < dev_size; ++i) {
    int length = dis_length_training(gen);
    int next_length = length;
    if (dis_negative(gen)) {
      while (next_length == length) { next_length = dis_length_training(gen); }
    }
    dev.emplace_back(length, next_length);
  }

  // Training.
  double loss = -std::numeric_limits<double>::infinity();
  for (size_t sample_num = 1; sample_num <= N; ++sample_num) {
    int length = dis_length_training(gen);
    int next_length = length;
    if (dis_negative(gen)) {
      while (next_length == length) { next_length = dis_length_training(gen); }
    }
    std::vector<std::shared_ptr<neural::Variable>> Xs;
    for (size_t i = 0; i < length; ++i) { Xs.push_back(model.MakeInput(i_a)); }
    for (size_t i = 0; i < next_length; ++i) {
      Xs.push_back(model.MakeInput(i_b));
    }
    auto h_last = rnn->Transduce(Xs).back().back()[0];
    auto W = model.MakeInput(i_W);
    auto h = W * h_last;
    auto l = binary_cross_entropy(h, {length == next_length});
    double new_loss = l->ForwardBackward();
    gd.UpdateWeights();
    loss = new_loss;

    if (verbose) {
      double prob = 1.0 / (1 + exp(-h->get_value(0)));
      bool correct = ((length == next_length && prob >= 0.5) ||
                      (length != next_length && prob < 0.5));
      std::string sample_string = "a^" + std::to_string(length) + " "
                                  "b^" + std::to_string(next_length);
      std::cout << "sample: " << sample_num << "     "
                << sample_string << "     "
                << "true=" << (length == next_length) << "     "
                << "correct=" << correct << "     "
                << "prob=" << prob << "     "
                << "loss: " << new_loss << std::endl;
    }

    if (sample_num % 1000 == 0) {
      size_t num_correct = 0;
      for (auto p : dev) {
        int length = p.first;
        int next_length = p.second;
        std::vector<std::shared_ptr<neural::Variable>> Xs;
        for (size_t i = 0; i < length; ++i) {
          Xs.push_back(model.MakeInput(i_a)); }
        for (size_t i = 0; i < next_length; ++i) {
          Xs.push_back(model.MakeInput(i_b));
        }
        auto W = model.MakeInput(i_W);
        auto h = W * rnn->Transduce(Xs).back().back()[0];
        auto l = binary_cross_entropy(h, {length == next_length});
        double prob = 1.0 / (1 + exp(-h->Forward()(0, 0)));
        bool correct = ((length == next_length && prob >= 0.5) ||
                        (length != next_length && prob < 0.5));
        if (correct) { ++num_correct; }
        model.ClearComputation();
      }
      std::cout << "at sample " << sample_num << ", dev acc  "
                << num_correct * 100.0 / dev.size() << std::endl;
    }
  }

  // Test.
  Eigen::MatrixXd example_states;
  std::string example_string;
  std::uniform_int_distribution<> dis_length_test(1, test_length);
  std::uniform_int_distribution<> dis_perturb(-2, 2);
  size_t num_correct = 0;
  size_t test_size = 1000;
  for (size_t test_num = 1; test_num <= test_size; ++test_num) {
    int n = dis_length_test(gen);
    int length = std::max(n + dis_perturb(gen), 1);
    int next_length = std::max(n + dis_perturb(gen), 1);

    std::vector<std::shared_ptr<neural::Variable>> Xs;
    for (size_t i = 0; i < length; ++i) {
      Xs.push_back(model.MakeInput(i_a)); }
    for (size_t i = 0; i < next_length; ++i) {
      Xs.push_back(model.MakeInput(i_b));
    }

    auto W = model.MakeInput(i_W);
    auto HHs = rnn->Transduce(Xs);
    auto h = W * HHs.back().back()[0];
    auto l = binary_cross_entropy(h, {length == next_length});
    double prob = 1.0 / (1 + exp(-h->Forward()(0, 0)));
    bool correct = ((length == next_length && prob >= 0.5) ||
                    (length != next_length && prob < 0.5));
    if (correct) { ++num_correct; }

    if (length + next_length >= example_states.rows() &&
        length + next_length <= 27) {
      example_string = "a^" + std::to_string(length) + " "
                       "b^" + std::to_string(next_length);
      example_states.resize(hdim, HHs.size());
      for (size_t t = 0; t < HHs.size(); ++t) {
        const auto &HH = (rnn_type == "lstm") ?
                         HHs[t].back()[1] : HHs[t].back()[0];
        Eigen::MatrixXd state_t_value = HH->ref_value();
        for (size_t i = 0; i < state_t_value.rows(); ++i) {
          example_states(i, t) = state_t_value(i);
        }
      }
    }

    model.ClearComputation();

    if (verbose) {
      std::string sample_string = "a^" + std::to_string(length) + " "
                                  "b^" + std::to_string(next_length);
      std::cout << "test: " << test_num << "     "
                << sample_string << "     "
                << "true=" << (length == next_length) << "     "
                << "correct=" << correct << "     "
                << "prob=" << prob << std::endl;
    }
  }
  std::cout << "test accuracy with sample size " << test_size
            << " and length " << test_length << ": "
            << num_correct * 100.0 / test_size << std::endl;

  std::cout << example_string << std::endl;
  std::cout << example_states << std::endl;
}
