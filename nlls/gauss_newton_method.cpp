#include "iostream"
#include "tceres/loss_function.h"
#include "tceres/problem.h"
#include "tceres/solver.h"

int main(int argc, char **argv) {
  tceres::Problem problem;
  tceres::LossFunction *loss_function = new tceres::HuberLoss(1.0);
  double out[3] = {0.0, 0., 0.};
  loss_function->Evaluate(1.0, out);
  std::cout << out[0] << std::endl;
  std::cout << "hello world\n";
  return 0;
}