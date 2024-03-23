#include "baproblem.hpp"

using namespace Eigen;
using namespace std;

constexpr int USE_POSE_SIZE = 6;

int main(int argc, const char *argv[])
{
    if (argc < 2)
    {
        cout << endl;
        cout << "Please type: " << endl;
        cout << "ba_demo [PIXEL_NOISE] " << endl;
        cout << endl;
        cout << "PIXEL_NOISE: noise in image space (E.g.: 1)" << endl;
        cout << endl;
        exit(0);
    }

    google::InitGoogleLogging(argv[0]);

    double PIXEL_NOISE = atof(argv[1]);

    cout << "PIXEL_NOISE: " << PIXEL_NOISE << endl;

    BAProblem<USE_POSE_SIZE> baProblem(15, 300, PIXEL_NOISE, true);

    tceres::Solver::Options options;
    options.minimizer_progress_to_stdout = true;
    options.trust_region_strategy_type = tceres::LEVENBERG_MARQUARDT;
    options.linear_solver_type = tceres::DENSE_SCHUR;
    options.minimizer_type = tceres::TRUST_REGION;
    tceres::Solver::Summary summary;
    baProblem.solve(options, &summary);
    std::cout << summary.BriefReport() << "\n";
    for (int i = 0; i < baProblem.states.poseNum; i++) {
        Eigen::Quaterniond Q_opt;
        Eigen::Vector3d t_opt;
        baProblem.states.getPose(i, Q_opt, t_opt);

        Eigen::Vector3d init_position = baProblem.before_opt_pose.at(i).t;
        std::cout << "idx: " << i
                  << ": init position: " << init_position.transpose()
                  << ", opt position: " << t_opt.transpose() << std::endl;
    }

    for (int i = 0; i < baProblem.states.pointNum; i++) {
        Eigen::Vector3d gt_pt, noise_pt, opt_pt;
        opt_pt = Eigen::Map<Eigen::Vector3d>(baProblem.states.point(i));
        gt_pt = Eigen::Map<Eigen::Vector3d>(baProblem.true_states.point(i));
        noise_pt = baProblem.noise_points.at(i);
        std::cout << "gt: " << gt_pt.transpose()
                  << ", noise pt: " << noise_pt.transpose()
                  << ", opt: " << opt_pt.transpose() << std::endl;
    }
}
