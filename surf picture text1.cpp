#include <opencv2/opencv.hpp>
//9#include <opencv2/xfeatures2d.hpp>
//#include <opencv2/xfeatures2d/nonfree.hpp>
#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>


#include <opencv2/opencv.hpp>
#include <opencv2/features2d/nonfree.hpp>
#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>
using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;

// 提速核心参数
const int SURF_HESSIAN_THRESHOLD = 400;
const int MAX_MATCHES = 100;
const double GOOD_MATCH_PERCENT = 0.15;

Mat stitchImagesFast(const Mat& img1, const Mat& img2) {
    // 1. 初始化SURF检测器（适配OpenCV 4.10.0）
    Ptr<SURF> detector = SURF::create(SURF_HESSIAN_THRESHOLD);
    detector->setUpright(true);
    detector->setNOctaves(3);
    detector->setNOctaveLayers(2);

    // 2. 灰度图检测特征点
    Mat gray1, gray2;
    cvtColor(img1, gray1, COLOR_BGR2GRAY);
    cvtColor(img2, gray2, COLOR_BGR2GRAY);

    vector<KeyPoint> keypoints1, keypoints2;
    Mat descriptors1, descriptors2;
    detector->detectAndCompute(gray1, noArray(), keypoints1, descriptors1);
    detector->detectAndCompute(gray2, noArray(), keypoints2, descriptors2);

    // 3. FLANN快速匹配
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
    vector<vector<DMatch>> knn_matches;
    matcher->knnMatch(descriptors1, descriptors2, knn_matches, 2);

    // 4. 筛选优质匹配点
    vector<DMatch> good_matches;
    for (size_t i = 0; i < knn_matches.size(); i++) {
        if (knn_matches[i][0].distance < GOOD_MATCH_PERCENT * knn_matches[i][1].distance) {
            good_matches.push_back(knn_matches[i][0]);
        }
    }
    if (good_matches.size() > MAX_MATCHES) {
        good_matches.resize(MAX_MATCHES);
    }

    // 5. 计算单应性矩阵
    vector<Point2f> points1, points2;
    for (size_t i = 0; i < good_matches.size(); i++) {
        points1.push_back(keypoints1[good_matches[i].queryIdx].pt);
        points2.push_back(keypoints2[good_matches[i].trainIdx].pt);
    }
    Mat homography = findHomography(points2, points1, RANSAC, 5.0);

    // 6. 透视变换+拼接
    Mat result;
    warpPerspective(img2, result, homography, Size(img1.cols + img2.cols, img1.rows), 
                    INTER_LINEAR, BORDER_TRANSPARENT);
    img1.copyTo(result(Rect(0, 0, img1.cols, img1.rows)));

    return result;
}

int main() {
    Mat img1 = imread("C:\\Users\\27722\\Desktop\\jjjtp\\2.jpg", 1);
    Mat img2 = imread("C:\\Users\\27722\\Desktop\\jjjtp\\111.jpg", 1);

    if (img1.empty() || img2.empty()) {
        cout << "错误：读取图像失败，请检查路径！" << endl;
        waitKey(0);
        return -1;
    }

    // 可选：缩小图像提速
    resize(img1, img1, Size(), 0.5, 0.5, INTER_NEAREST);
    resize(img2, img2, Size(), 0.5, 0.5, INTER_NEAREST);

    double start = getTickCount();
    Mat stitched = stitchImagesFast(img1, img2);
    double duration = (getTickCount() - start) / getTickFrequency();
    cout << "拼接耗时：" << duration << " 秒" << endl;

    imshow("SURF拼接结果", stitched);
    imwrite("C:\\Users\\27722\\Desktop\\jjjtp\\surf_stitch_result.jpg", stitched);
    waitKey(0);
    destroyAllWindows();

    return 0;
}