#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <iostream>
#include <vector>
#include <algorithm>
//比text3 快一点 视频
using namespace cv;
using namespace std;

// 拼接角点结构体
typedef struct {
    Point2f left_top;
    Point2f left_bottom;
    Point2f right_top;
    Point2f right_bottom;
} four_corners_t;

four_corners_t corners;

// 计算透视变换后的四角点
void CalcCorners(const Mat& H, const Mat& src) {
    double v2[3], v1[3];
    Mat V2(3, 1, CV_64FC1, v2), V1(3, 1, CV_64FC1, v1);

    // 左上角 (0,0,1)
    v2[0] = 0; v2[1] = 0; v2[2] = 1;
    V1 = H * V2;
    corners.left_top = Point2f(v1[0]/v1[2], v1[1]/v1[2]);

    // 左下角 (0,src.rows,1)
    v2[0] = 0; v2[1] = src.rows; v2[2] = 1;
    V1 = H * V2;
    corners.left_bottom = Point2f(v1[0]/v1[2], v1[1]/v1[2]);

    // 右上角 (src.cols,0,1)
    v2[0] = src.cols; v2[1] = 0; v2[2] = 1;
    V1 = H * V2;
    corners.right_top = Point2f(v1[0]/v1[2], v1[1]/v1[2]);

    // 右下角 (src.cols,src.rows,1)
    v2[0] = src.cols; v2[1] = src.rows; v2[2] = 1;
    V1 = H * V2;
    corners.right_bottom = Point2f(v1[0]/v1[2], v1[1]/v1[2]);
}

// 快速拼接（简化版，无过度融合，优先速度）
void FastStitch(Mat& img_left, Mat& img_right_warp, Mat& dst) {
    // 确定拼接后图像的尺寸
    int dst_w = max(img_right_warp.cols, img_left.cols);
    int dst_h = max(img_right_warp.rows, img_left.rows);
    dst.create(dst_h, dst_w, CV_8UC3);
    dst.setTo(0);

    // 拷贝右图像（透视变换后）
    img_right_warp.copyTo(dst(Rect(0, 0, img_right_warp.cols, img_right_warp.rows)));
    // 拷贝左图像（覆盖重叠区域）
    img_left.copyTo(dst(Rect(0, 0, img_left.cols, img_left.rows)));
}

int main() {
    // -------------------------- 1. 初始化（移到循环外，避免重复创建） --------------------------
    // 视频路径（替换为你的视频路径，避免中文/长路径）
    string video_path1 = "C:\\jjjtp\\dog2.mp4";  // 右视频
    string video_path2 = "C:\\jjjtp\\dog1.mp4";  // 左视频
    VideoCapture cap1(video_path1), cap2(video_path2);
    
    if (!cap1.isOpened() || !cap2.isOpened()) {
        cerr << "视频打开失败！请检查路径是否正确" << endl;
        return -1;
    }

    // 帧缩放尺寸（降采样提速，可根据视频分辨率调整）
    const int SCALE_WIDTH = 640;
    const int SCALE_HEIGHT = 480;

    // ORB特征检测器（全局初始化，复用）
    Ptr<ORB> orb = ORB::create(1000, 1.2f, 8, 31, 0, 2, ORB::HARRIS_SCORE, 31, 20);
    flann::SearchParams search_params(50);  // 搜索迭代次数（越小越快）

    // 内存复用容器（避免循环内频繁申请）
    Mat frame1, frame2, gray1, gray2;
    Mat frame1_scaled, frame2_scaled, gray1_scaled, gray2_scaled;
    vector<KeyPoint> kp1, kp2;
    Mat desc1, desc2;
    Mat matchIndex, matchDistance;
    vector<DMatch> good_matches;
    vector<Point2f> pts1, pts2;
    Mat homo, warp_frame1, dst;

    // -------------------------- 2. 视频循环拼接 --------------------------
    while (true) {
        double t_start = getTickCount();

        // 读取帧
        cap1 >> frame1;
        cap2 >> frame2;
        if (frame1.empty() || frame2.empty()) {
            cout << "视频读取完毕/帧为空" << endl;
            break;
        }

        // 降采样（核心提速点）
        resize(frame1, frame1_scaled, Size(SCALE_WIDTH, SCALE_HEIGHT));
        resize(frame2, frame2_scaled, Size(SCALE_WIDTH, SCALE_HEIGHT));
        // 转灰度
        cvtColor(frame1_scaled, gray1_scaled, COLOR_BGR2GRAY);
        cvtColor(frame2_scaled, gray2_scaled, COLOR_BGR2GRAY);

        // -------------------------- 3. 特征检测+匹配（高效逻辑） --------------------------
        // 检测ORB特征（复用检测器）
        orb->detectAndCompute(gray1_scaled, Mat(), kp1, desc1);
        orb->detectAndCompute(gray2_scaled, Mat(), kp2, desc2);

        if (desc1.empty() || desc2.empty() || kp1.size() < 10 || kp2.size() < 10) {
            cout << "特征点过少，跳过当前帧" << endl;
            continue;
        }

        // ✅ 修复点2：FLANN索引直接内嵌LshIndexParams参数（你要求的极简写法）
        flann::Index flann_index(desc1, flann::LshIndexParams(12, 20, 2), cvflann::FLANN_DIST_HAMMING);
        matchIndex.create(desc2.rows, 2, CV_32SC1);
        matchDistance.create(desc2.rows, 2, CV_32FC1);
        flann_index.knnSearch(desc2, matchIndex, matchDistance, 2, search_params);

        // 筛选优质匹配（Lowe算法，阈值0.6平衡精度/速度）
        good_matches.clear();
        for (int i = 0; i < matchDistance.rows; i++) {
            float dist1 = matchDistance.at<float>(i, 0);
            float dist2 = matchDistance.at<float>(i, 1);
            if (dist1 < 0.6 * dist2) {
                good_matches.emplace_back(i, matchIndex.at<int>(i, 0), dist1);
            }
        }

        if (good_matches.size() < 8) {  // 至少8个点计算单应性矩阵
            cout << "匹配点过少（<8），跳过当前帧" << endl;
            continue;
        }

        // -------------------------- 4. 单应性矩阵计算 --------------------------
        pts1.clear(); pts2.clear();
        for (const auto& m : good_matches) {
            pts1.push_back(kp1[m.trainIdx].pt);
            pts2.push_back(kp2[m.queryIdx].pt);
        }
        // RANSAC抗噪（重投影误差阈值2.0，平衡鲁棒性/速度）
        homo = findHomography(pts1, pts2, RANSAC, 2.0);
        if (homo.empty()) {
            cout << "单应性矩阵计算失败，跳过当前帧" << endl;
            continue;
        }

        // -------------------------- 5. 透视变换+快速拼接 --------------------------
        CalcCorners(homo, frame1_scaled);
        // 计算透视变换后的尺寸（仅取有效区域，减少计算量）
        int warp_w = max((int)corners.right_top.x, (int)corners.right_bottom.x);
        warpPerspective(frame1_scaled, warp_frame1, homo, Size(warp_w, frame1_scaled.rows));
        // 快速拼接（无过度融合，优先速度）
        FastStitch(frame2_scaled, warp_frame1, dst);

        // -------------------------- 6. 显示+帧率输出 --------------------------
        imshow("拼接结果", dst);
        imshow("左视频", frame2_scaled);
        imshow("右视频", frame1_scaled);

        // 计算帧率
        double fps = getTickFrequency() / (getTickCount() - t_start);
        cout << "帧率: " << fps << " | 有效匹配点数: " << good_matches.size() << endl;

        // ESC键（27）退出，waitKey(1)保证实时性
        if (waitKey(1) == 27) break;
    }

    // 释放资源
    cap1.release();
    cap2.release();
    destroyAllWindows();
    return 0;
}