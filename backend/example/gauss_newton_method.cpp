// Copyright (C) 2007 Free Software Foundation, Inc. <https://fsf.org/>
// Everyone is permitted to copy and distribute verbatim copies
// of this license document, but changing it is not allowed.
// This file is part of vio-hello-world Copyright (C) 2023 ZJU
// You should have received a copy of the GNU General Public License
// along with vio-hello-world. If not, see <https://www.gnu.org/licenses/>.
// Author: weihao(isweihao@zju.edu.cn), M.S at Zhejiang University

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