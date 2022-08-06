#include <perspectiveNPoint.h>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <opencv2/opencv.hpp>

#include <ctime>
clock_t startTime, endTime;

int main() {
    // 初始化测试
    std::cout << "Perspective-n-Point Lib Test" << std::endl;
    PerspectiveNPointClass PerspectiveNPoint;

    // 构造 3D 点云
    std::vector<Eigen::Vector3d> points3D;
    std::vector<cv::Point3f> points3D_cv;
    for (int i = 1; i < 10; i++) {
        for (int j = 1; j < 10; j++) {
            for (int k = 1; k < 10; k++) {
                points3D.emplace_back(Eigen::Vector3d(i, j, k * 2.0));
                points3D_cv.emplace_back(cv::Point3f(i, j, k * 2.0));
            }
        }
    }

    // 定义相机位姿（相对于世界坐标系）
    Eigen::Matrix3d R_cw = Eigen::Matrix3d::Identity();
    Eigen::Vector3d t_cw = Eigen::Vector3d(1, -3, 0);

    // 将 3D 点云通过两帧位姿映射到对应的归一化平面上，构造匹配点对
    std::vector<Eigen::Vector3d> points2D;
    std::vector<cv::Point2f> points2D_cv;
    for (unsigned long i = 0; i < points3D.size(); i++) {
        Eigen::Vector3d tempP = R_cw * points3D[i] + t_cw;
        points2D.emplace_back(Eigen::Vector3d(tempP(0, 0) / tempP(2, 0), tempP(1, 0) / tempP(2, 0), 1.0));
        points2D_cv.emplace_back(cv::Point2f(points2D.back().x(), points2D.back().y()));
    }

    // 随机给匹配点增加 outliers
    std::vector<int> outliersIndex;
    for (unsigned int i = 0; i < points3D.size() / 100; i++) {
        int idx = rand() % points3D.size();
        points2D[idx](0, 0) = 0;
        outliersIndex.emplace_back(idx);
    }

    // 定义相机内参
    cv::Mat K = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1 );
    Eigen::Matrix3d CameraMatrix;
    CameraMatrix << 1, 0, 0, 0, 1, 0, 0, 0, 1;
    double fx = CameraMatrix(0, 0);
    double fy = CameraMatrix(1, 1);
    double cx = CameraMatrix(0, 2);
    double cy = CameraMatrix(1, 2);
    std::cout << std::endl;

    /*--------------------------------------------------------------------------------------------------*/
    // 通过 PnP 方法求解相机位姿
    Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
    Eigen::Vector3d t = Eigen::Vector3d::Zero();
    std::vector<uchar> status;
    std::cout << "My code solve PnP:" << std::endl;
    startTime = clock();
    PerspectiveNPoint.EstimateRotationAndTranslation(points3D, points2D, R, t, status);
    endTime = clock();
    std::cout << "Time cost " << (double)(endTime - startTime) / CLOCKS_PER_SEC << std::endl;
    std::cout << R << std::endl << t << std::endl;
    std::cout << std::endl;

    // 采用 OpenCV 库来求解相机文字
    std::cout << "OpenCV solve PnP:" << std::endl;
    cv::Mat R_cv = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);
    cv::Mat rvec_cv;
    cv::Rodrigues(R_cv, rvec_cv);
    cv::Mat t_cv = (cv::Mat_<double>(3, 1) << 0, 0, 0);
    startTime = clock();
    cv::solvePnP(points3D_cv, points2D_cv, K, cv::Mat(), rvec_cv, t_cv);
    endTime = clock();
    std::cout << "Time cost " << (double)(endTime - startTime) / CLOCKS_PER_SEC << std::endl;
    cv::Rodrigues(rvec_cv, R_cv);
    std::cout << R_cv << std::endl << t_cv << std::endl;
    std::cout << std::endl;

    // 打印出真实的结果
    std::cout << "The real pose is:" << std::endl;
    std::cout << R_cw << std::endl << t_cw << std::endl;

    /*-----------------------------------------------------------------------------------------------------*/
    // 检查 status 是否符合预期
    status.clear();
    PerspectiveNPoint.EstimateRotationAndTranslation(points3D, points2D, R, t, status);
    std::cout << "outliers detect result:" << std::endl;
    for (unsigned int i = 0; i < outliersIndex.size(); i++) {
        std::cout << (int)status[outliersIndex[i]];
    }
    std::cout << "\ninliers detect result:" << std::endl;
    for (unsigned int i = 0; i < status.size(); i++) {
        if (i % 100 == 0 && i != 0) {
            std::cout << std::endl;
        }
        std::cout << (int)status[i];
    }
    std::cout << std::endl;

    return 0;
}
