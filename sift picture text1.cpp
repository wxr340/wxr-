#include <opencv2/opencv.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/calib3d.hpp>
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
using namespace cv::xfeatures2d;
using namespace std;

// 计算透视变换后的四角点，确定拼接图像尺寸
void calcCorners(const Mat& H, const Mat& src, Point2f& left_top, Point2f& left_bottom,
                 Point2f& right_top, Point2f& right_bottom) {
    double v2[3], v1[3];
    Mat V2 = Mat(3, 1, CV_64FC1, v2);
    Mat V1 = Mat(3, 1, CV_64FC1, v1);

    // 左上角 (0,0,1)
    v2[0] = 0; v2[1] = 0; v2[2] = 1;
    V1 = H * V2;
    left_top = Point2f(v1[0]/v1[2], v1[1]/v1[2]);

    // 左下角 (0,src.rows,1)
    v2[0] = 0; v2[1] = src.rows; v2[2] = 1;
    V1 = H * V2;
    left_bottom = Point2f(v1[0]/v1[2], v1[1]/v1[2]);

    // 右上角 (src.cols,0,1)
    v2[0] = src.cols; v2[1] = 0; v2[2] = 1;
    V1 = H * V2;
    right_top = Point2f(v1[0]/v1[2], v1[1]/v1[2]);

    // 右下角 (src.cols,src.rows,1)
    v2[0] = src.cols; v2[1] = src.rows; v2[2] = 1;
    V1 = H * V2;
    right_bottom = Point2f(v1[0]/v1[2], v1[1]/v1[2]);
}

// 图像拼接核心函数
bool stitchImages(const string& imgPath1, const string& imgPath2, const string& savePath) {
    // 1. 读取图像（img1：右图，img2：左图，需有重叠区域）
    Mat img1 = imread(imgPath1);
    Mat img2 = imread(imgPath2);
    if (img1.empty() || img2.empty()) {
        cerr << "错误：图像读取失败！请检查路径：" << imgPath1 << " 或 " << imgPath2 << endl;
        return false;
    }

    // 2. 转换为灰度图（SIFT特征提取需灰度输入）
    Mat gray1, gray2;
    cvtColor(img1, gray1, COLOR_BGR2GRAY);
    cvtColor(img2, gray2, COLOR_BGR2GRAY);

    // 3. SIFT特征提取（OpenCV 4.x需用xfeatures2d模块）
    Ptr<SIFT> sift = SIFT::create(15000); // 最多提取1000个特征点，平衡精度与速度
    vector<KeyPoint> kp1, kp2; // 特征点容器
    Mat desc1, desc2;           // 特征描述子容器
    sift->detectAndCompute(gray1, noArray(), kp1, desc1);
    sift->detectAndCompute(gray2, noArray(), kp2, desc2);
    cout << "SIFT特征提取完成：" << endl;
    cout << "  - 右图特征点数量：" << kp1.size() << endl;
    cout << "  - 左图特征点数量：" << kp2.size() << endl;

    // 4. 暴力匹配（BFMatcher，NORM_L2适配SIFT的浮点型描述子）
    BFMatcher matcher(NORM_L2);
    vector<vector<DMatch>> knnMatches; // KNN匹配结果（取Top2）
    matcher.knnMatch(desc1, desc2, knnMatches, 2);

    // 5. 筛选优质匹配（Lowe算法，阈值0.75剔除误匹配）
    vector<DMatch> goodMatches;
    for (size_t i = 0; i < knnMatches.size(); i++) {
        if (knnMatches[i][0].distance < 0.75 * knnMatches[i][1].distance) {
            goodMatches.push_back(knnMatches[i][0]);
        }
    }
    cout << "优质匹配点数量：" << goodMatches.size() << endl;
    if (goodMatches.size() < 8) { // 至少8个点才能计算单应性矩阵
        cerr << "错误：优质匹配点不足8个，无法完成拼接！" << endl;
        return false;
    }

    // 6. 提取匹配点的坐标
    vector<Point2f> pts1, pts2;
    for (const auto& match : goodMatches) {
        pts1.push_back(kp1[match.queryIdx].pt);   // 右图匹配点
        pts2.push_back(kp2[match.trainIdx].pt);   // 左图匹配点
    }

    // 7. 计算单应性矩阵（RANSAC算法抗噪，重投影误差阈值2.0）
    Mat homo = findHomography(pts1, pts2, RANSAC, 2.0);
    if (homo.empty()) {
        cerr << "错误：单应性矩阵计算失败！" << endl;
        return false;
    }

    // 8. 计算透视变换后的图像尺寸
    Point2f left_top, left_bottom, right_top, right_bottom;
    calcCorners(homo, img1, left_top, left_bottom, right_top, right_bottom);
    int warpWidth = max((int)right_top.x, (int)right_bottom.x); // 变换后右图的宽度
    int warpHeight = img1.rows;                                 // 高度与原图一致

    // 9. 对右图执行透视变换（对齐到左图坐标系）
    Mat warpImg1;
    warpPerspective(img1, warpImg1, homo, Size(warpWidth, warpHeight));

    // 10. 拼接图像（左图覆盖到变换后的右图重叠区域）
    Mat resultImg = Mat::zeros(Size(warpWidth, max(warpHeight, img2.rows)), CV_8UC3);
    warpImg1.copyTo(resultImg(Rect(0, 0, warpImg1.cols, warpImg1.rows))); // 拷贝变换后的右图
    img2.copyTo(resultImg(Rect(0, 0, img2.cols, img2.rows)));             // 拷贝左图覆盖重叠区

    // 定义缩放函数（等比例调整尺寸0）
    auto resizeImage = [](const cv::Mat& img, int max_width = 800, int max_height = 600) -> cv::Mat {
        cv::Mat resized_img;
        int img_w = img.cols;
        int img_h = img.rows;
        // 计算等比例缩放因子（取宽/高中较小的比例，不超最大尺寸）
        double scale = std::min((double)max_width / img_w, (double)max_height / img_h);
        if (scale < 1.0) { // 仅当图像超过最大尺寸时缩放
            cv::resize(img, resized_img, cv::Size(), scale, scale, cv::INTER_AREA);
        } else {
            resized_img = img.clone(); // 尺寸合适则直接复制
        }
        return resized_img;
    };

    // 调整各图像尺寸后显示（可自定义max_width/max_height）
    cv::Mat resized_img1 = resizeImage(img1, 1800, 1600);       // 右图缩放（最大800x600）
    cv::Mat resized_img2 = resizeImage(img2, 1800, 1600);       // 左图缩放（最大800x600）
    cv::Mat resized_result = resizeImage(resultImg, 11200, 1800); // 拼接结果缩放（最大1200x800）

    
    // 显示调整尺寸后的图像
    imshow("拼接结果（已缩放）", resized_result);
    imshow("右图（待变换）", resized_img1);
    imshow("左图（基准）", resized_img2);
    cout << "拼接完成！结果已保存至：" << savePath << endl;

    waitKey(0); // 按任意键关闭窗口
    destroyAllWindows();
    return true;
}

int main() {
    // ========== 不要中文/空格） ==========
     string imgPath1 = "C:\\jjjtp\\fangziy.png";  // 右图（透视变换的图）
     string imgPath2 = "C:\\jjjtp\\fangziz.png";   // 左图（基准图）
     string savePath = "C:\\images\\stitch_result.jpg"; // 结果保存路径

    
    bool success = stitchImages(imgPath1, imgPath2, savePath);
    return success ? 0 : -1;
}