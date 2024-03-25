#include "backend/cost_function/reprojection_se3_factor.hpp"
#include "backend/parameter_block/point_parameter_block.hpp"
#include "backend/parameter_block/se3_parameter_block.hpp"
#include "backend/parameterization/se3left_parametrization.hpp"
#include "fstream"
#include "gflags/gflags.h"
#include "opencv2/core/eigen.hpp"
#include "opencv2/opencv.hpp"
#include "tceres/loss_function.h"
#include "tceres/problem.h"
#include "tceres/solver.h"

DEFINE_string(pts_3d_file_path, "../backend/data/p3d.txt",
              "3d point file path");
DEFINE_string(pts_2d_file_path, "../backend/data/p2d.txt",
              "2d point file path");

void ReadPoint2dDataset(const std::string &p2d_file,
                        const std::string &p3d_file,
                        std::vector<Eigen::Vector3d> &pts_3d_vec,
                        std::vector<Eigen::Vector2d> &pts_2d_vec,
                        bool print_output) {
  std::ifstream fp3d(p3d_file);
  std::ifstream fp2d(p2d_file);
  if (!fp3d || !fp2d) {
    std::cerr << "Error Open file of fp3d or fp2d!\n";
    return;
  }
  while (!fp3d.eof()) {
    double pt3d[3] = {0.0};
    for (auto &p : pt3d) {
      fp3d >> p;
    }
    Eigen::Vector3d point3d;
    point3d << pt3d[0], pt3d[1], pt3d[2];
    pts_3d_vec.push_back(point3d);
  }
  while (!fp2d.eof()) {
    double pt2d[2] = {0.0};
    for (auto &p : pt2d) {
      fp2d >> p;
    }
    Eigen::Vector2d point2d;
    point2d << pt2d[0], pt2d[1];
    pts_2d_vec.push_back(point2d);
  }
  if (print_output) {
    std::for_each(pts_3d_vec.begin(), pts_3d_vec.end(),
                  [](const Eigen::Vector3d &pts_3d) -> void {
                    std::cout << pts_3d.transpose() << "; \t";
                  });
    std::for_each(pts_2d_vec.begin(), pts_2d_vec.end(),
                  [](const Eigen::Vector2d &pts_2d) -> void {
                    std::cout << pts_2d.transpose() << "; \t";
                  });
  }
}

bool tceresPNP(const std::vector<Eigen::Vector2d> &vunkps,
               const std::vector<Eigen::Vector3d> &vwpts,
               const std::vector<int> &vscales, Sophus::SE3d &Twc,
               const int nmaxiter, const float chi2th, const bool buse_robust,
               const bool bapply_l2_after_robust, const float fx,
               const float fy, const float cx, const float cy,
               std::vector<int> &voutliersidx) {
  assert(vunkps.size() == vwpts.size());

  tceres::Problem problem;

  double chi2thrtsq = std::sqrt(chi2th);

  tceres::LossFunctionWrapper *loss_function;
  loss_function = new tceres::LossFunctionWrapper(
      new tceres::HuberLoss(chi2thrtsq), tceres::TAKE_OWNERSHIP);

  if (!buse_robust) {
    loss_function->Reset(nullptr, tceres::TAKE_OWNERSHIP);
  }

  size_t nbkps = vunkps.size();

  tceres::LocalParameterization *local_parameterization =
      new backend::SE3LeftParameterization();

  backend::PoseParametersBlock posepar = backend::PoseParametersBlock(0, Twc);

  problem.AddParameterBlock(posepar.values(), 7, local_parameterization);

  std::vector<backend::DirectLeftSE3::ReprojectionErrorSE3 *> verrors_;
  std::vector<tceres::ResidualBlockId> vrids_;

  for (size_t i = 0; i < nbkps; i++) {
    auto *f = new backend::DirectLeftSE3::ReprojectionErrorSE3(
        vunkps[i].x(), vunkps[i].y(), fx, fy, cx, cy, vwpts.at(i),
        std::pow(2., vscales[i]));

    tceres::ResidualBlockId rid =
        problem.AddResidualBlock(f, loss_function, posepar.values());

    verrors_.push_back(f);
    vrids_.push_back(rid);
  }

  tceres::Solver::Options options;
  options.linear_solver_type = tceres::DENSE_SCHUR;
  options.trust_region_strategy_type = tceres::LEVENBERG_MARQUARDT;

  options.num_threads = 1;
  options.max_num_iterations = nmaxiter;
  options.max_solver_time_in_seconds = 0.005;
  options.function_tolerance = 1.e-3;

  options.minimizer_progress_to_stdout = false;

  tceres::Solver::Summary summary;
  tceres::Solve(options, &problem, &summary);
  Twc = posepar.getPose();
  return summary.IsSolutionUsable();
}

bool opencvP3PRansac(const std::vector<Eigen::Vector3d> &bvs,
                     const std::vector<Eigen::Vector3d> &vwpts,
                     const int nmaxiter, const float errth, const float fx,
                     const float fy, const bool boptimize, Sophus::SE3d &Twc,
                     std::vector<int> &voutliersidx) {
  assert(bvs.size() == vwpts.size());

  size_t nb3dpts = bvs.size();

  if (nb3dpts < 4) {
    return false;
  }

  voutliersidx.reserve(nb3dpts);

  std::vector<cv::Point2f> cvbvs;
  cvbvs.reserve(nb3dpts);

  std::vector<cv::Point3f> cvwpts;
  cvwpts.reserve(nb3dpts);

  for (size_t i = 0; i < nb3dpts; i++) {
    cvbvs.push_back(cv::Point2f(bvs.at(i).x(), bvs.at(i).y()));

    cvwpts.push_back(
        cv::Point3f(vwpts.at(i).x(), vwpts.at(i).y(), vwpts.at(i).z()));
  }

  // Using homoegeneous pts here so no dist or calib.
  cv::Mat D;
  cv::Mat K = cv::Mat::eye(3, 3, CV_32F);

  cv::Mat tvec, rvec;
  cv::Mat inliers;

  bool use_extrinsic_guess = false;
  float confidence = 0.99;

  float focal = (fx + fy) / 2.;

  bool status = cv::solvePnPRansac(cvwpts, cvbvs, K, D, rvec, tvec,
                                   use_extrinsic_guess, nmaxiter, errth / focal,
                                   confidence, inliers, cv::SOLVEPNP_ITERATIVE);
  if (inliers.rows == 0) {
    return false;
  }

  int k = 0;
  for (size_t i = 0; i < nb3dpts; i++) {
    if (inliers.at<int>(k) == (int)i) {
      k++;
      if (k == inliers.rows) {
        k = 0;
      }
    } else {
      voutliersidx.push_back(i);
    }
  }

  if (voutliersidx.size() >= nb3dpts - 5) {
    return false;
  }

  if (boptimize) {
    use_extrinsic_guess = true;

    // Filter outliers
    std::vector<cv::Point2f> in_cvbvs;
    in_cvbvs.reserve(inliers.rows);

    std::vector<cv::Point3f> in_cvwpts;
    in_cvwpts.reserve(inliers.rows);

    for (int i = 0; i < inliers.rows; i++) {
      in_cvbvs.push_back(cvbvs.at(inliers.at<int>(i)));
      in_cvwpts.push_back(cvwpts.at(inliers.at<int>(i)));
    }

    cv::solvePnP(in_cvwpts, in_cvbvs, K, D, rvec, tvec, use_extrinsic_guess,
                 cv::SOLVEPNP_ITERATIVE);
  }

  // Store the resulting pose
  cv::Mat cvR;
  cv::Rodrigues(rvec, cvR);

  Eigen::Vector3d tcw;
  Eigen::Matrix3d Rcw;

  cv::cv2eigen(cvR, Rcw);
  cv::cv2eigen(tvec, tcw);

  Twc.translation() = -1. * Rcw.transpose() * tcw;
  Twc.setRotationMatrix(Rcw.transpose());

  return true;
}

std::vector<Eigen::Vector3d> pixelConvert(std::vector<Eigen::Vector2d> &p2d,
                                          const float fx, const float fy,
                                          const float cx, const float cy) {
  Eigen::Matrix3d K;
  std::vector<Eigen::Vector3d> bv_pts;
  K << fx, 0, cx, 0, fy, cy, 0, 0, 1;
  for (const auto &i : p2d) {
    Eigen::Vector3d pt(i.x(), i.y(), 1);
    bv_pts.emplace_back(K.inverse() * pt);
  }
  return bv_pts;
}

int main(int argc, char **argv) {
  google::ParseCommandLineFlags(&argc, &argv, true);
  std::vector<Eigen::Vector3d> p3d;
  std::vector<Eigen::Vector2d> p2d;

  const std::string pts_3d_file = FLAGS_pts_3d_file_path;
  const std::string pts_2d_file = FLAGS_pts_2d_file_path;
  ReadPoint2dDataset(pts_2d_file, pts_3d_file, p3d, p2d, false);
  std::printf("size 3d: %ld, size 2d: %ld\n", p3d.size(), p2d.size());
  std::vector<int> vscales(p2d.size(), 0);
  std::vector<int> voutliersidx;
  float K[4] = {520.9, 521.0, 325.1, 249.7};  // fx,fy,cx,cy
  Sophus::SE3d pose_ceres, pose_cv;
  bool ceres_solver_status =
      tceresPNP(p2d, p3d, vscales, pose_ceres, 5, 5.9915, false, false, K[0],
                K[1], K[2], K[3], voutliersidx);

  std::vector<Eigen::Vector3d> bv_pts =
      pixelConvert(p2d, K[0], K[1], K[2], K[3]);
  bool opencv_solver_status = opencvP3PRansac(
      bv_pts, p3d, 5, 5.9915, K[0], K[1], false, pose_cv, voutliersidx);
  std::printf("opencv solver status: %d\n", opencv_solver_status);
  std::printf("ceres  solver status: %d\n", ceres_solver_status);
  std::cout << "ceres solver result:\n"
            << pose_ceres.inverse().matrix3x4() << std::endl;
  std::cout << "opencv solver result:\n"
            << pose_cv.inverse().matrix3x4() << std::endl;
  double err = (pose_ceres * pose_cv.inverse()).log().norm();
  std::printf("solver error: %f\n", err);
  return 0;
}