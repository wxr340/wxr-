#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d.hpp>
#include <iostream>
#include <vector>
#include <algorithm>

using namespace cv;
using namespace std;

// 计算透视变换后的四角点（用于确定画布尺寸）
void calcCorners(const Mat& H, const Mat& src, std::vector<Point2f>& corners) {
    corners.clear();
    double v2[3], v1[3];
    cv::Mat V2 = cv::Mat(3, 1, CV_64FC1, v2);
    cv::Mat V1 = cv::Mat(3, 1, CV_64FC1, v1);

    std::vector<Point2f> src_corners = {
        Point2f(0, 0),
        Point2f(0, static_cast<float>(src.rows)),
        Point2f(static_cast<float>(src.cols), 0),
        Point2f(static_cast<float>(src.cols), static_cast<float>(src.rows))
    };

    for (const auto& pt : src_corners) {
        v2[0] = pt.x;
        v2[1] = pt.y;
        v2[2] = 1.0;
        V1 = H * V2;
        corners.push_back(Point2f(static_cast<float>(v1[0]/v1[2]), static_cast<float>(v1[1]/v1[2])));
    }
}

// 等比例缩放图像（解决显示过大问题）
Mat resizeImage(const Mat& img, int max_width = 800, int max_height = 600) {
    Mat resized_img;
    if (img.empty()) return resized_img;
    int img_w = img.cols;
    int img_h = img.rows;
    double scale = std::min(static_cast<double>(max_width) / img_w, static_cast<double>(max_height) / img_h);
    if (scale < 1.0) {
        resize(img, resized_img, Size(), scale, scale, INTER_AREA);
    } else {
        resized_img = img.clone();
    }
    return resized_img;
}

// ORB算法实现图像拼接（核心函数，含羽化融合解决缝合线）
bool orbImageStitch(const std::string& imgPath1, const std::string& imgPath2, const std::string& savePath) {
    // 1. 读取图像（img1：右图，img2：左图，需有重叠区域）
    Mat img1 = imread(imgPath1, cv::IMREAD_COLOR);
    Mat img2 = imread(imgPath2, cv::IMREAD_COLOR);
    if (img1.empty() || img2.empty()) {
        std::cerr << "错误：图像读取失败！请检查路径：" << imgPath1 << " 或 " << imgPath2 << std::endl;
        return false;
    }

    // 2. 转换为灰度图（ORB特征提取需灰度输入）
    Mat gray1, gray2;
    cvtColor(img1, gray1, COLOR_BGR2GRAY);
    cvtColor(img2, gray2, COLOR_BGR2GRAY);

    // 3. ORB特征提取初始化
    Ptr<ORB> orb = ORB::create(
        6000,    // 增加特征点数量 改的越大 拼的越好
        1.2f,    // 金字塔缩放因子
        8,       // 金字塔层数
        31,      // 角点检测邻域大小
        0,       // 边缘阈值
        2,       // WTA_K
        ORB::HARRIS_SCORE, // Harris评分筛选优质角点
        31,      // 描述子邻域大小
        20       // 最小距离阈值
    );

    // 4. 检测特征点并计算描述子
    std::vector<KeyPoint> kp1, kp2;
    Mat desc1, desc2; // ORB输出8位二进制描述子
    orb->detectAndCompute(gray1, noArray(), kp1, desc1);
    orb->detectAndCompute(gray2, noArray(), kp2, desc2);

    std::cout << "ORB特征提取完成：" << std::endl;
    std::cout << "  - 右图特征点数量：" << kp1.size() << std::endl;
    std::cout << "  - 左图特征点数量：" << kp2.size() << std::endl;

    // 5. ORB暴力匹配（二进制描述子用NORM_HAMMING距离）
    BFMatcher matcher(NORM_HAMMING);
    std::vector<std::vector<DMatch>> knnMatches;
    matcher.knnMatch(desc1, desc2, knnMatches, 2); // KNN取Top2

    // 6. 筛选优质匹配（Lowe算法，阈值0.75）
    std::vector<DMatch> goodMatches;
    for (size_t i = 0; i < knnMatches.size(); i++) {
        if (knnMatches[i][0].distance < 0.75 * knnMatches[i][1].distance) {
            goodMatches.push_back(knnMatches[i][0]);
        }
    }

    std::cout << "优质匹配点数量：" << goodMatches.size() << std::endl;
    if (goodMatches.size() < 8) { // 单应性矩阵最低要求8个匹配点
        std::cerr << "错误：优质匹配点不足8个，无法完成拼接！" << std::endl;
        return false;
    }

    // 7. 提取匹配点坐标
    std::vector<Point2f> pts1, pts2;
    for (const auto& match : goodMatches) {
        pts1.push_back(kp1[match.queryIdx].pt);   // 右图匹配点
        pts2.push_back(kp2[match.trainIdx].pt);   // 左图匹配点
    }

    // 8. 计算单应性矩阵（RANSAC抗噪，重投影误差2.0）
    Mat homo = findHomography(pts1, pts2, RANSAC, 2.0);
    if (homo.empty()) {
        std::cerr << "错误：单应性矩阵计算失败！" << std::endl;
        return false;
    }

    // 9. 计算画布尺寸（解决ROI越界核心逻辑）
    std::vector<Point2f> transformed_corners;
    calcCorners(homo, img1, transformed_corners);

    // 计算所有角点的极值（覆盖右图变换后+左图区域）
    float min_x = std::min({transformed_corners[0].x, transformed_corners[1].x, transformed_corners[2].x, transformed_corners[3].x, 0.0f});
    float min_y = std::min({transformed_corners[0].y, transformed_corners[1].y, transformed_corners[2].y, transformed_corners[3].y, 0.0f});
    float max_x = std::max({transformed_corners[0].x, transformed_corners[1].x, transformed_corners[2].x, transformed_corners[3].x, static_cast<float>(img2.cols)});
    float max_y = std::max({transformed_corners[0].y, transformed_corners[1].y, transformed_corners[2].y, transformed_corners[3].y, static_cast<float>(img2.rows)});

    // 画布最终尺寸
    int canvas_width = static_cast<int>(std::ceil(max_x - min_x));
    int canvas_height = static_cast<int>(std::ceil(max_y - min_y));

    // 偏移矩阵（处理负坐标，避免越界）
    Mat offset_mat = Mat::eye(3, 3, CV_64FC1);
    if (min_x < 0) offset_mat.at<double>(0, 2) = -min_x;
    if (min_y < 0) offset_mat.at<double>(1, 2) = -min_y;

    // 10. 透视变换（叠加偏移矩阵）
    Mat final_homo = offset_mat * homo;
    Mat warpImg1;
    warpPerspective(img1, warpImg1, final_homo, Size(canvas_width, canvas_height));

    // 11. 拼接图像（羽化融合解决缝合线）
    // 计算左图拷贝位置，确保ROI不越界
    int img2_x = (min_x < 0) ? static_cast<int>(-min_x) : 0;
    int img2_y = (min_y < 0) ? static_cast<int>(-min_y) : 0;
    Rect img2_roi = Rect(
        img2_x,
        img2_y,
        std::min(img2.cols, canvas_width - img2_x),
        std::min(img2.rows, canvas_height - img2_y)
    );

    // ====== 彻底重写的羽化融合核心逻辑 ======
    // 1. 先把左图拷贝到结果画布上
    Mat resultImg = Mat::zeros(canvas_height, canvas_width, CV_8UC3);
    img2.copyTo(resultImg(img2_roi));

    // 2. 找出右图（warpImg1）的有效区域（非全黑）
    Mat warpImg1_gray;
    cvtColor(warpImg1, warpImg1_gray, COLOR_BGR2GRAY);
    Mat warpMask = warpImg1_gray > 0; // 有效区域为白色(255)，无效为黑色(0)

    // 3. 找到重叠区域的边界
    std::vector<std::vector<Point>> contours;
    findContours(warpMask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    if (contours.empty()) {
        std::cerr << "错误：未检测到有效区域！" << std::endl;
        return false;
    }
    Rect overlapRect = boundingRect(contours[0]); // 有效区域的外接矩形

    // 4. 创建羽化掩码（在重叠区域做线性渐变）
    Mat mask = Mat::zeros(warpImg1.size(), CV_32FC1);
    for (int y = overlapRect.y; y < overlapRect.y + overlapRect.height; y++) {
        for (int x = overlapRect.x; x < overlapRect.x + overlapRect.width; x++) {
            if (warpMask.at<uchar>(y, x) == 255) {
                // 计算x方向的渐变权重，越靠右，右图权重越低
                double ratio = static_cast<double>(x - overlapRect.x) / static_cast<double>(overlapRect.width);
                mask.at<float>(y, x) = static_cast<float>(1.0 - ratio);
            }
        }
    }

    // 5. 加权融合
    Mat warpImg1_float, result_float;
    warpImg1.convertTo(warpImg1_float, CV_32FC3);
    resultImg.convertTo(result_float, CV_32FC3);

    // 右图 × 掩码
    multiply(warpImg1_float, mask, warpImg1_float);
    // 左图 × (1 - 掩码)
    Mat inv_mask = 1.0 - mask;
    multiply(result_float, inv_mask, result_float);
    // 叠加
    result_float += warpImg1_float;

    // 6. 转回8位图像
    result_float.convertTo(resultImg, CV_8UC3);
    // =====================================

    // 12. 保存+缩放显示结果
    if (!imwrite(savePath, resultImg)) {
        std::cerr << "错误：结果保存失败！请检查路径：" << savePath << std::endl;
        return false;
    }

    // 缩放显示
    Mat resized_img1 = resizeImage(img1, 800, 600);
    Mat resized_img2 = resizeImage(img2, 800, 600);
    Mat resized_result = resizeImage(resultImg, 1200, 800);

    if (!resized_img1.empty()) imshow("右图（待变换）", resized_img1);
    if (!resized_img2.empty()) imshow("左图（基准）", resized_img2);
    if (!resized_result.empty()) imshow("ORB算法拼接结果（已缩放）", resized_result);
    std::cout << "拼接完成！结果已保存至：" << savePath << std::endl;

    waitKey(0);
    destroyAllWindows();
    return true;
}

// 主函数：替换路径即可运行
int main() {
    // ========== 替换为你的本地图像路径（绝对路径，无中文/空格） ==========
    std::string imgPath1 = "C:\\jjjtp\\szy.jpg";  // 右图（需要透视变换的图）
    std::string imgPath2 = "C:\\jjjtp\\szz.jpg";   // 左图（基准图，无需变换）
    std::string savePath = "C:\\images\\stitch_result.jpg"; // 拼接结果保存路径

    // 执行ORB图像拼接
    bool success = orbImageStitch(imgPath1, imgPath2, savePath);
    return success ? 0 : -1;
}