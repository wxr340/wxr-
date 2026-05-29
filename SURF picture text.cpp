#include <opencv2/opencv.hpp>
//#include <opencv2/xfeatures2d.hpp>
#include <iostream>
#include <vector>


#include <opencv2/opencv.hpp>
#include <opencv2/features2d/nonfree.hpp>
#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>
using namespace cv;
using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;

// 可选算法：SIFT/SURF/ORB，改这里就能切换
#define USE_ALGORITHM SURF  // 可选：SIFT / SURF / ORB

// 提速参数
const int HESSIAN_THRESHOLD = 1500;  // SURF/SIFT用
const int ORB_FEATURE_NUM = 500;    // ORB用
const int MAX_MATCHES = 1000;
const double GOOD_MATCH_PERCENT = 0.7;



// 通用拼接函数（适配所有算法）
Mat stitchImages(const Mat& img1, const Mat& img2) {
    Ptr<Feature2D> detector;
    // 根据选择的算法初始化检测器
#if USE_ALGORITHM == SIFT
    detector = SIFT::create(HESSIAN_THRESHOLD);
#elif USE_ALGORITHM == SURF
    detector = SURF::create(HESSIAN_THRESHOLD);
    dynamic_cast<SURF*>(detector.get())->setUpright(true); // 提速
#elif USE_ALGORITHM == ORB
    detector = ORB::create(ORB_FEATURE_NUM);
#endif

    // 灰度图检测特征点
    Mat gray1, gray2;
    cvtColor(img1, gray1, COLOR_BGR2GRAY);
    cvtColor(img2, gray2, COLOR_BGR2GRAY);

    vector<KeyPoint> kp1, kp2;
    Mat desc1, desc2;
    detector->detectAndCompute(gray1, noArray(), kp1, desc1);
    detector->detectAndCompute(gray2, noArray(), kp2, desc2);

    // FLANN匹配
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
    vector<vector<DMatch>> knn_matches;
    matcher->knnMatch(desc1, desc2, knn_matches, 2);

    // 筛选匹配点
    vector<DMatch> good_matches;
    for (size_t i = 0; i < knn_matches.size(); i++) {
        if (knn_matches[i][0].distance < GOOD_MATCH_PERCENT * knn_matches[i][1].distance) {
            good_matches.push_back(knn_matches[i][0]);
        }
    }
    if (good_matches.size() > MAX_MATCHES) good_matches.resize(MAX_MATCHES);

    // 计算单应性矩阵
    vector<Point2f> pts1, pts2;
    for (auto& m : good_matches) {
        pts1.push_back(kp1[m.queryIdx].pt);
        pts2.push_back(kp2[m.trainIdx].pt);
    }
    Mat homography = findHomography(pts2, pts1, RANSAC, 5.0);

    // 拼接图像
    Mat result;
    warpPerspective(img2, result, homography, Size(img1.cols + img2.cols, img1.rows), INTER_LINEAR, BORDER_TRANSPARENT);
    img1.copyTo(result(Rect(0, 0, img1.cols, img1.rows)));

    return result;
}

int main() {
    // 替换为你的图片路径
    Mat img1 = imread("C:\\jjjtp\\szz.jpg", 1);//zuo
    Mat img2 = imread("C:\\jjjtp\\szy.jpg", 1);//you

    if (img1.empty() || img2.empty()) {
        cout << "错误：读取图片失败！" << endl;
        waitKey(0);
        return -1;
    }

    // 缩小图像提速
    resize(img1, img1, Size(), 0.5, 0.5);
    resize(img2, img2, Size(), 0.5, 0.5);

    // 拼接+计时
    double start = getTickCount();
    Mat stitched = stitchImages(img1, img2);
    cout << "拼接耗时：" << (getTickCount() - start)/getTickFrequency() << "秒" << endl;

    // 显示+保存
    imshow("拼接结果", stitched);
    imwrite("C:\\Users\\27722\\Desktop\\jjjtp\\stitch_result.jpg", stitched);
    waitKey(0);
    destroyAllWindows();

    return 0;
}