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
  //  / load training dataset
  std::vector<TestCase> train_data;
  printf("load train data...\t");
  LoadTestCases("../examples/mnist_train.csv", train_data, 60'000);
  printf("done \n");

  printf("train data set size: %d\n", train_data.size());

  TApplication app("app", &argc, argv);
  //  DisplayTestCase(test_data[12], app);

  auto mg = TGraph();

  NeuralNet nn(784, {64, 4, 64}, 784);
  //  nn.GetActivationFunction(-4) = ActivationFunction::SIGMOID;
  //  nn.GetActivationFunction(-3) = ActivationFunction::SIGMOID;
  //  nn.GetActivationFunction(-2) = ActivationFunction::SIGMOID;
  //  nn.GetActivationFunction(-1) = ActivationFunction::SIGMOID;
  nn.FillRandom();

  double learning_rate = 0.1;

  const int kEpochs = 10;
  const int kMiniBatchSize = 10;

  TestCase one;
  int k = 0;

  auto t1 = std::chrono::system_clock::now();

  for (int e = 0; e < kEpochs; e++) {
    double average_epoch_error = 0.0;

    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::shuffle(train_data.begin(), train_data.end(),
                 std::default_random_engine(seed));

    std::cout << "e: " << e
              << std::chrono::duration_cast<std::chrono::seconds>(
                     std::chrono::system_clock::now() - t1)
                     .count()
              << " seconds\n";
    for (int b = 0; b < train_data.size(); b++) {

      double error_sum = 0;
      Nabla nabla;

      nn.FeedForward(train_data[b].GetInput());
      auto error = nn.PowCostFunction(train_data[b].GetInput());
      nabla += nn.PropagateBackwards(error);
      error_sum += Sum(error);
      average_epoch_error += Sum(error);

      nn.Update(nabla, learning_rate);

      if (b % 100 == 99) {
        mg.SetPoint(k, k, error_sum / (double)kMiniBatchSize);

        k += 1;
      }
    }
  }
  printf("done! \a \n");
  nn.SaveToFile("../examples/MNIST-encoder/mnist-encoder");

  auto c = new TCanvas("canvas", "NeuralNets", 10, 10, 800, 600);
  mg.SetTitle("Global_Net_Error;Iterations;Error");
  mg.SetMarkerStyle(22);
  mg.SetFillStyle(0);
  mg.SetMarkerSize(0);
  mg.SetDrawOption("LP");
  mg.SetLineColor(4);
  mg.SetLineWidth(2);
  mg.Draw();

  TRootCanvas *rc = (TRootCanvas *)c->GetCanvasImp();
  app.Run();

  return 0;
}