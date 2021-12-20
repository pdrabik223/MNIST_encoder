//
// Created by piotr on 19/12/2021.
//
//
// Created by piotr on 19/12/2021.
//

#include "matrix.h"
#include "neural_net.h"
#include <chrono>
#include <fstream>
#include <iostream>

#include "TApplication.h"
#include "TCanvas.h"
#include "TF1.h"
#include "TGraph.h"

#include "TH2F.h"

#include "TRootCanvas.h"
#include <random>

class TestCase {
public:
  TestCase() : input(784, 1), label(0) {}
  TestCase(const std::vector<int> &input_values, int label_val)
      : input(784, 1), label(label_val) {

    for (int i = 0; i < 28; i++)
      for (int j = 0; j < 28; j++)
        input.Get((28 - i - 1) * 28 + j) =
            (double)input_values[i * 28 + j] / 255.0;
  }

  // {}
  const matrix::Matrix<double> &GetInput() const { return input; }
  const int &GetLabel() const { return label; }

protected:
  /// 784 values from 0 to 1
  matrix::Matrix<double> input;
  int label;
};

void LoadTestCases(const std::string &csv_file_path,
                   std::vector<TestCase> &target, int no_test_cases = 0) {
  target.clear();
  std::ifstream file(csv_file_path);
  std::string line;
  std::getline(file, line);
  if (no_test_cases == 0)
    while (file.good()) {
      int label;
      file >> label;
      char coma;
      file >> coma;
      std::vector<int> pixels;
      for (int i = 0; i < 784; i++) {
        int pixel;
        file >> pixel;
        if (i < 783)
          file >> coma;
        pixels.push_back(pixel);
      }
      target.emplace_back(pixels, label);
    }
  else
    for (int t = 0; t < no_test_cases; t++) {
      int label;
      file >> label;
      char coma;
      file >> coma;
      std::vector<int> pixels;
      for (int i = 0; i < 784; i++) {
        int pixel;
        file >> pixel;
        if (i < 783)
          file >> coma;
        pixels.push_back(pixel);
      }
      target.emplace_back(pixels, label);
    }
  file.close();
}

int main(int argc, char **argv) {
  TApplication app("app", &argc, argv);

  std::vector<TestCase> test_data;
  printf("load test data...\t");
  LoadTestCases("../examples/mnist_test.csv", test_data, 10'000);

  printf("done \n");
  printf("test data set size: %d", test_data.size());

  NeuralNet nn("../examples/MNIST-encoder/mnist-encoder");

  double average_net_error = 0.0;
  for (int i = 0; i < 10'000; i++) {
    nn.FeedForward(test_data[i].GetInput());
    auto error = nn.CostFunction(test_data[i].GetInput());
    average_net_error += Sum(error);
  }
  printf("average net error: %lf", average_net_error / 10'000.0);

  int frame_width = 28;
  int frame_height = 28;
  auto c = new TCanvas("canvas", "NeuralNets", 10, 10, 800, 600);

  std::vector<TH2F> images;
  for (int i = 0; i < 8; i++) {

    int test = rand() % test_data.size();

    images.emplace_back(std::string("h2_i" + std::to_string(test)).c_str(),
                        "test", frame_width, 0, frame_width, frame_height, 0,
                        frame_height);

    for (auto x = 0; x < frame_width; x++)
      for (auto y = 0; y < frame_height; y++) {
        images.back().Fill(
            y, x, test_data[test].GetInput().Get((x * frame_width) + y) + 0.1);
      }

    std::string label = "Label: " + std::to_string(test_data[test].GetLabel());

    images.back().SetStats(false);
    images.back().SetContour(255);
    images.back().SetTitle(label.c_str());
    images.back().SetFillStyle(0);

    images.emplace_back(std::string("h2_o" + std::to_string(test)).c_str(),
                        "test", frame_width, 0, frame_width, frame_height, 0,
                        frame_height);

    auto output = nn.FeedForward(test_data[test].GetInput());

    for (auto x = 0; x < frame_width; x++)
      for (auto y = 0; y < frame_height; y++) {
        images.back().Fill(y, x, output.Get((x * frame_width) + y) + 0.1);
      }

    label = "net output: " + std::to_string(test_data[test].GetLabel());

    images.back().SetStats(false);
    images.back().SetContour(255);
    images.back().SetTitle(label.c_str());
    images.back().SetFillStyle(0);
  }

  c->Divide(4, 4);
  for (int i = 0; i < 16; i++) {
    c->cd(i + 1);
    images[i].Draw("colz");
  }

  TRootCanvas *rc = (TRootCanvas *)c->GetCanvasImp();
  app.Run();

  return 0;
}
