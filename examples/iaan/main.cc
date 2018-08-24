// Author: Karl Stratos (me@karlstratos.com)

#include <ctime>
#include <limits>
#include <iostream>
#include <stdlib.h>
#include <string>
#include <vector>

#include "../../core/neural.h"
#include "../../core/eval.h"

int main (int argc, char* argv[]) {
  std::string model_path;
  std::string words_path;
  std::string tags_path;
  bool train = false;
  size_t zsize = 45;
  size_t wdim = 100;
  size_t cdim = 0;
  size_t width = 2;
  size_t batch_size = 80;
  double step_size = 0.01;
  size_t num_epochs = 10;
  size_t random_seed = 42;

  if (util_misc::want_options(argc, argv)) {
    std::cout << "model:\t path to model directory" << std::endl;
    std::cout << "words:\t text data (one sentence per line)" << std::endl;
    std::cout << "tags:\t gold tags (one tag sequence per line)" << std::endl;
    std::cout << std::endl;

    std::cout << "--train:             \t"
              << "train a model?" << std::endl;
    std::cout << "--zsize [" << zsize << "]:       \t"
              << "number of tags" << std::endl;
    std::cout << "--wdim [" << wdim << "]:        \t"
              << "dimension of word embeddings" << std::endl;
    std::cout << "--cdim [" << cdim << "]:        \t"
              << "dimension of character embeddings" << std::endl;
    std::cout << "--width [" << width << "]:        \t"
              << "context width" << std::endl;
    std::cout << "--batch [" << batch_size << "]:        \t"
              << "batch size" << std::endl;
    std::cout << "--step [" << step_size << "]:        \t"
              << "step size" << std::endl;
    std::cout << "--epochs [" << num_epochs << "]:        \t"
              << "number of epochs" << std::endl;
    std::cout << "--seed [" << random_seed << "]:        \t"
              << "random seed" << std::endl;
    exit(0);
  }
  model_path = argv[1];
  words_path = argv[2];
  tags_path = argv[3];
  for (int i = 4; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--train") {
      train = true;
    } else if (arg == "--zsize") {
      zsize = std::stoi(argv[++i]);
    } else if (arg == "--wdim") {
      wdim = std::stoi(argv[++i]);
    } else if (arg == "--cdim") {
      cdim = std::stoi(argv[++i]);
    } else if (arg == "--width") {
      width = std::stoi(argv[++i]);
    } else if (arg == "--batch") {
      batch_size = std::stoi(argv[++i]);
    } else if (arg == "--step") {
      step_size = std::stod(argv[++i]);
    } else if (arg == "--epochs") {
      num_epochs = std::stoi(argv[++i]);
    } else if (arg == "--seed") {
      random_seed = std::stoi(argv[++i]);
    } else {
      std::cerr << "Invalid argument \"" << arg << "\": run the command with "
                << "-h or --help to see possible arguments." << std::endl;
      exit(-1);
    }
  }
  std::srand(random_seed);

  neural::Model model;
  auto word_sequences = util_file::read_lines(words_path);
  auto tag_sequences = util_file::read_lines(tags_path);

  auto w2i = util_misc::build_dictionary(word_sequences);
  auto c2i = util_misc::build_character_dictionary(word_sequences);
  size_t i_Ew = model.AddWeight(wdim, w2i.size(), "unit-variance");
  size_t i_buffer = model.AddWeight(wdim, 1, "unit-variance");
  size_t i_Ec = model.AddWeight(cdim, c2i.size(), "unit-variance");
  neural::LSTM clstm1(1, cdim, cdim, &model);
  neural::LSTM clstm2(1, cdim, cdim, &model);
  size_t i_WX = model.AddWeight(zsize, 2 * wdim * width, "unit-variance");
  size_t i_WY = model.AddWeight(zsize, 2 * cdim + wdim, "unit-variance");

  auto prepare_batches = [&]() {
    std::vector<std::pair<size_t, size_t>> pairs;
    for (size_t i = 0; i < word_sequences.size(); ++i) {
      for (size_t j = 0; j < word_sequences[i].size(); ++j) {
        pairs.push_back(std::make_pair(i, j));
      }
    }
    std::shuffle(pairs.begin(), pairs.end(),
                 std::default_random_engine(random_seed));
    return util_misc::segment(pairs, batch_size);
  };

  auto get_fseq = [&](const std::string &word) {
    std::vector<std::shared_ptr<neural::Variable>> fseq;
    for (char c : word) {
      fseq.push_back(model.MakeInputColumn(i_Ec, c2i[c]));
    }
    return fseq;
  };

  auto get_bseq = [&](const std::string &word) {
    std::vector<std::shared_ptr<neural::Variable>> bseq;
    for (int i = word.size() - 1; i >= 0; --i) {
      bseq.push_back(model.MakeInputColumn(i_Ec, c2i[word[i]]));
    }
    return bseq;
  };

  auto get_word_emb = [&](const std::string &word) {
    return model.MakeInputColumn(i_Ew, w2i[word]);
  };

  auto get_context_emb = [&](size_t sentence_index, size_t word_index) {
    const auto &sentence = word_sequences[sentence_index];
    std::vector<std::shared_ptr<neural::Variable>> wembs;
    for (int i = static_cast<int>(word_index) - static_cast<int>(width);
         i < static_cast<int>(word_index); ++i) {
      wembs.push_back((i >= 0) ? model.MakeInputColumn(i_Ew, w2i[sentence[i]]) :
                      model.MakeInput(i_buffer));
    }
    for (size_t i = word_index + 1; i <= word_index + width; ++i) {
      wembs.push_back((i < sentence.size()) ?
                      model.MakeInputColumn(i_Ew, w2i[sentence[i]]) :
                      model.MakeInput(i_buffer));
    }
    return vcat(wembs);
  };

  auto get_loss = [&](const std::vector<std::pair<size_t, size_t>> &batch) {
    auto WX = model.MakeInput(i_WX);
    auto WY = model.MakeInput(i_WY);
    std::vector<std::vector<std::shared_ptr<neural::Variable>>> fseqs;
    std::vector<std::vector<std::shared_ptr<neural::Variable>>> bseqs;
    std::vector<std::shared_ptr<neural::Variable>> word_embs;
    std::vector<std::shared_ptr<neural::Variable>> context_embs;
    for (const auto &ij : batch) {
      if (cdim > 0) {
        fseqs.push_back(get_fseq(word_sequences[ij.first][ij.second]));
        bseqs.push_back(get_bseq(word_sequences[ij.first][ij.second]));
      }
      word_embs.push_back(get_word_emb(word_sequences[ij.first][ij.second]));
      context_embs.push_back(get_context_emb(ij.first, ij.second));
    }
    std::shared_ptr<neural::Variable> pZ_Ys;
    if (cdim > 0) {
      auto freps = hcat(clstm1.EncodeByFinalTop(fseqs));
      auto breps = hcat(clstm2.EncodeByFinalTop(bseqs));
      std::vector<std::shared_ptr<neural::Variable>> fbw = {freps, breps,
                                                            hcat(word_embs)};
      pZ_Ys = softmax(WY * vcat(fbw));
    } else {
      pZ_Ys = softmax(WY * hcat(word_embs));
    }
    auto qZ_Xs = softmax(WX * hcat(context_embs));
    auto pZ = average_rwise(pZ_Ys);
    auto hZ = entropy(pZ, true);
    auto hZ_X = average(cross_entropy(pZ_Ys, qZ_Xs, true));
    auto loss = hZ_X - hZ;
    return loss;
  };

  auto compute_performance = [&]() {
    std::vector<std::vector<std::string>> pred_sequences;
    for (size_t i = 0; i < word_sequences.size(); ++i) {
      std::vector<std::string> pred_sequence;
      auto WY = model.MakeInput(i_WY);
      for (size_t j = 0; j < word_sequences[i].size(); ++j) {
        const std::string &word = word_sequences[i][j];
        Eigen::MatrixXd probabilities;
        auto wemb = get_word_emb(word);
        if (cdim > 0) {
          auto femb = clstm1.EncodeByFinalTop({get_fseq(word)})[0];
          auto bemb = clstm2.EncodeByFinalTop({get_bseq(word)})[0];
          std::vector<std::shared_ptr<neural::Variable>> fbw = {femb, bemb,
                                                                wemb};
          probabilities = softmax(WY * vcat(fbw))->Forward();
        } else {
          probabilities = softmax(WY * wemb)->Forward();
        }
        Eigen::MatrixXd::Index z;
        probabilities.col(0).maxCoeff(&z);
        pred_sequence.push_back(std::to_string(z));
      }
      pred_sequences.push_back(pred_sequence);
      model.ClearComputation();
    }
    return eval::compute_many2one_accuracy(tag_sequences, pred_sequences).first;
  };

  neural::Adam gd(&model, step_size);
  for (size_t epoch_num = 1; epoch_num <= num_epochs; ++epoch_num) {
    auto batches = prepare_batches();
    double total_batch_loss = 0.0;
    size_t k = 0;
    for (const auto &batch : batches) {
      ++k;
      //std::cout << "batch " << k << " / " << batches.size() << std::endl;
      auto loss = get_loss(batch);
      double loss_value = loss->ForwardBackward();
      gd.UpdateWeights();
      total_batch_loss += loss_value;
    }
    double avg_loss = total_batch_loss  / batches.size();
    std::cout << "Epoch " << util_string::buffer_string(
        std::to_string(epoch_num), 3, ' ', "right")
              << "  updates: " << batches.size()
              << "  loss: " << util_string::printf_format("%.2f", avg_loss)
              << std::endl;
    /*
    double perf = compute_performance();
    std::cout << "Epoch " << util_string::buffer_string(
        std::to_string(epoch_num), 3, ' ', "right")
              << "  updates: " << batches.size()
              << "  loss: " << util_string::printf_format("%.2f", avg_loss)
              << "  Y perf: " << util_string::printf_format("%.2f", perf)
              << std::endl;
    */
  }
}
