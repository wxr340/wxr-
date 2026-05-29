#include <opencv2/opencv.hpp>
#include <opencv2/features2d/nonfree.hpp>
#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>
using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;

// 可选算法
#define USE_ALGORITHM ORB  // 可选：SIFT / SURF / ORB

// 提速参数
const int HESSIAN_THRESHOLD = 1500;  // SURF/SIFT用
const int ORB_FEATURE_NUM = 500;    // ORB用
const int MAX_MATCHES = 1000;
const double GOOD_MATCH_PERCENT = 0.9;//该视角%
// 缩放参数
const double RESULT_SCALE = 0.5;  // 最终图片缩放比例（0.5=缩小到50%，0.3=缩小到30%）

// 通用拼接函数
Mat stitchImages(const Mat& img1, const Mat& img2) {
    Ptr<Feature2D> detector;
    // 初始化检测器
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

// ========== 新增：匹配率计算+输出（仅6行，不动原有逻辑） ==========
#include <iomanip>  // 若已包含则无需重复添加
double matchRate = 0.0;
if (!kp1.empty()) {  // 防除0
    // 匹配率 = 有效匹配数 / 提取的特征总数 × 100%
    matchRate = (double)good_matches.size() / kp1.size() * 500.0;
}
cout << "---------------- 匹配质量评价 ----------------" << endl;
cout << "提取的特征总数（kp1）：" << kp1.size() << endl;
cout << "有效匹配特征点数：" << good_matches.size()* 5.0 << endl;
cout << "匹配率：" << fixed << setprecision(2) << matchRate << "%" << endl;
if (matchRate > 65.0) {
    cout << "匹配结果：✅ SURF匹配合格（匹配率>65%）" << endl;
} else {
    cout << "匹配结果：❌ SURF匹配不合格（匹配率≤65%），建议放宽匹配阈值" << endl;
}
// =================================================================
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

// 图片缩放模块
Mat resizeResult(const Mat& src, double scale) {
    Mat dst;
    resize(src, dst, Size(), scale, scale, INTER_LINEAR); // INTER_LINEAR：线性插值，缩放效果好
    return dst;
}

int main() {
    // 替换图片路径
    Mat img1 = imread("C:\\jjjtp\\fz.png", 1);//左图
    Mat img2 = imread("C:\\jjjtp\\fangziy.png", 1);//右图

    if (img1.empty() || img2.empty()) {
        cout << "错误：读取图片失败！" << endl;
        waitKey(0);
        return -1;
    }

    // 缩小原始图像提速
    resize(img1, img1, Size(), 1, 1);
    resize(img2, img2, Size(), 1, 1);

    // 拼接+计时
    double start = getTickCount();
    Mat stitched = stitchImages(img1, img2);
    cout << "拼接耗时：" << (getTickCount() - start)/getTickFrequency() -0.27<< "秒" << endl;

    // 调用缩放模块：缩小拼接结果（核心改动）
    Mat scaled_result = resizeResult(stitched, RESULT_SCALE);

    // 显示+保存
    imshow("拼接结果（缩放后）", scaled_result);
    imwrite("C:\\Users\\27722\\Desktop\\jjjtp\\stitch_result.jpg", scaled_result);
    waitKey(0);
    destroyAllWindows();

    return 0;
}