//
// Created by piotr on 10/12/2021.
//

#include "neural_net.h"
#include <chrono>
#include <fstream>
#include <iostream>
#include "matrix.h"

#include "TApplication.h"
#include "TCanvas.h"
#include "TF1.h"
#include "TGraph.h"
#include "TH1.h"
#include "TH2F.h"

#include "TRootCanvas.h"
#include <random>

// double Sum(matrix::Matrix<double> &target) {
//   double sum = 0.0;
//   for (auto i : target.RawData())
//     sum += abs(i);
//   return sum;
// }

struct TestCase {
  TestCase() : input(784, 1), label(10, 1) {}
  TestCase(const std::vector<int> &input_values, int label_val)
      : input(784, 1), label(10, 1) {
    for (int i = 0; i < 28; i++)
      for (int j = 0; j < 28; j++)
        input.Get((28 - i - 1) * 28 + j) =
            (double)input_values[i * 28 + j] / 255.0;

    for (int i = 0; i < 10; i++)
      if (i == label_val)
        label.Get(i) = 1.0;
      else
        label.Get(i) = 0.0;
  }

  /// 784 values from 0 to 1
  matrix::Matrix<double> input;
  /// 10 values only one is 1
  matrix::Matrix<double> label;
};
double Max(const matrix::Matrix<double> &target) {
  double max = -100000.0;

  for (double i : target.GetData()) {
    if (i > max)
      max = i;
  }
  return max;
}
int MaxId(const matrix::Matrix<double> &target) {
  int id = 0;
  for (int i = 0; i < target.GetHeight(); i++)
    if (target.Get(i) > target.Get(id))
      id = i;
  return id;
}

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


  NeuralNet nn("../MNIST-try/mnistNN");
  printf("id\tlabel\tnn approximation\n");

  printf("==============test data============\n");
  for (int i = 0; i < 10; i++) {

    //    printf("%d\n", i);
    std::cout << "label:    \t"
              << ToString(matrix::Transpose(test_data[i].label)) << "\n";
    std::cout << "net aprox:\t"
              << ToString(matrix::Transpose(nn.FeedForward(test_data[i].input)))
              << "\n";
  }
  double average_net_error = 0.0;
  for (int i = 0; i < 10'000; i++) {

    auto error = nn.CostFunction(test_data[i].label);
    average_net_error += Sum(error);
  }
  printf("average net error: %lf", average_net_error / 10'000.0);

  int frame_width = 28;
  int frame_height = 28;
  auto c = new TCanvas("canvas", "NeuralNets", 10, 10, 800, 600);

  std::vector<TH2F> images;
  for (int i = 0; i < 9; i++) {

    int test = rand() % test_data.size();

    images.emplace_back(std::string("h2" + std::to_string(test)).c_str(),
                        "test", frame_width, 0, frame_width, frame_height, 0,
                        frame_height);

    images.back().SetStats(false);
    images.back().SetContour(255);

    for (auto x = 0; x < frame_width; x++)
      for (auto y = 0; y < frame_height; y++) {
        images.back().Fill(
            y, x, test_data[test].input.Get((x * frame_width) + y) + 0.1);
      }

    auto net_output = nn.FeedForward(test_data[test].input);
    auto error = nn.CostFunction(test_data[test].label);
    double skalar_error = Sum(error);
    std::string label =
        "Label: " + std::to_string(MaxId(test_data[test].label)) +
        " net estimation: " + std::to_string(MaxId(net_output)) +
        " output value: " + std::to_string(Max(net_output)) +
        " error: " + std::to_string(skalar_error);
    images.back().SetTitle(label.c_str());
    images.back().SetFillStyle(0);
  }

  c->Divide(3, 3);
  for (int i = 0; i < 9; i++) {
    c->cd(i + 1);
    images[i].Draw("colz");
  }

  TRootCanvas *rc = (TRootCanvas *)c->GetCanvasImp();
  app.Run();

  return 0;
}
void DisplayMugolStyle(NeuralNet &nn, const std::vector<TestCase> &test_data) {}
