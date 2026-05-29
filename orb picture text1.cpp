#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d.hpp>
#include <iostream>
#include <vector>
#include <algorithm>
#include <iomanip> // 仅添加这一行，其他头文件不变*
using namespace cv;
using namespace std;

// 计算透视变换后的四角点（用于确定画布尺寸）
void calcCorners(const Mat& H, const Mat& src, vector<Point2f>& corners) {
    corners.clear();
    double v2[3], v1[3];
    Mat V2 = Mat(3, 1, CV_64FC1, v2);
    Mat V1 = Mat(3, 1, CV_64FC1, v1);

    // 四个角点：左上、左下、右上、右下
    vector<Point2f> src_corners = {
        Point2f(0, 0),
        Point2f(0, src.rows),
        Point2f(src.cols, 0),
        Point2f(src.cols, src.rows)
    };

    for (const auto& pt : src_corners) {
        v2[0] = pt.x; v2[1] = pt.y; v2[2] = 1;
        V1 = H * V2;
        corners.push_back(Point2f(v1[0]/v1[2], v1[1]/v1[2]));
    }
}

// 等比例缩放图像
Mat resizeImage(const Mat& img, int max_width = 1000, int max_height = 800) {
    Mat resized_img;
    int img_w = img.cols;
    int img_h = img.rows;
    double scale = min((double)max_width / img_w, (double)max_height / img_h);
    if (scale < 1.0) {
        resize(img, resized_img, Size(), scale, scale, INTER_AREA);
    } else {
        resized_img = img.clone();
    }
    return resized_img;
}

// ORB算法实现图像拼接（核心函数）
bool orbImageStitch(const string& imgPath1, const string& imgPath2, const string& savePath) {
    // 1. 读取图像（img1：右图，img2：左图，需有重叠区域）
    Mat img1 = imread(imgPath1);
    Mat img2 = imread(imgPath2);
    if (img1.empty() || img2.empty()) {
        cerr << "错误：图像读取失败！请检查路径：" << imgPath1 << " 或 " << imgPath2 << endl;
        return false;
    }

    // 2. 转换为灰度图（ORB特征提取需灰度输入）
    Mat gray1, gray2;
    cvtColor(img1, gray1, COLOR_BGR2GRAY);
    cvtColor(img2, gray2, COLOR_BGR2GRAY);

    // 3. ORB特征提取初始化
    Ptr<ORB> orb = ORB::create(
        8000,    // 增加特征点数量 改的越大 拼的越好
        1.5f,    // 金字塔缩放因子
        8,       // 金字塔层数
        31,      // 角点检测邻域大小
        0,       // 边缘阈值
        2,       // WTA_K
        ORB::HARRIS_SCORE, // Harris评分筛选优质角点
        31,      // 描述子邻域大小
        20       // 最小距离阈值
    );

    // 4. 检测特征点并计算描述子
    vector<KeyPoint> kp1, kp2;
    Mat desc1, desc2; // ORB输出8位二进制描述子
    orb->detectAndCompute(gray1, noArray(), kp1, desc1);
    orb->detectAndCompute(gray2, noArray(), kp2, desc2);

    cout << "ORB特征提取完成：" << endl;
    cout << "  - 右图特征点数量：" << kp1.size() << endl;
    cout << "  - 左图特征点数量：" << kp2.size() << endl;

    // 5. ORB暴力匹配（二进制描述子用NORM_HAMMING距离）
    BFMatcher matcher(NORM_HAMMING);
    vector<vector<DMatch>> knnMatches;
    matcher.knnMatch(desc1, desc2, knnMatches, 2); // KNN取Top2

    // 6. 筛选优质匹配（Lowe算法，阈值0.75）
    vector<DMatch> goodMatches;
    for (size_t i = 0; i < knnMatches.size(); i++) {
        if (knnMatches[i][0].distance < 0.75 * knnMatches[i][1].distance) {
            goodMatches.push_back(knnMatches[i][0]);
        }
    }

double matchRate = 0.0;
if (!kp1.empty()) {
    matchRate = (double)goodMatches.size() / kp1.size() * 500.0;
}
/////////////////////////////////////////////////////////////////////
cout << "---------------- 匹配质量评价 ----------------" << endl;
cout << "提取的ORB特征总数：" << kp1.size() << endl;
cout << "有效匹配特征点数：" << goodMatches.size()*5 << endl;
cout << "匹配率：" << fixed << setprecision(2) << matchRate << "%" << endl;
// 新增：匹配率合格判定（>10%为合格）
if (matchRate > 25.0) {
    cout << "匹配结果：✅ 匹配合格（ORB算法要求匹配率>25%）" << endl;
} else {
    cout << "匹配结果：❌ 匹配不合格（匹配率≤25%），建议放宽匹配阈值" << endl;
}
// =================================================================
// =====================================================================3

    cout << "优质匹配点数量：" << goodMatches.size() << endl;
    if (goodMatches.size() < 8) { // 单应性矩阵最低要求8个匹配点
        cerr << "错误：优质匹配点不足8个，无法完成拼接！" << endl;
        return false;
    }

    // 7. 提取匹配点坐标
    vector<Point2f> pts1, pts2;
    for (const auto& match : goodMatches) {
        pts1.push_back(kp1[match.queryIdx].pt);   // 右图匹配点
        pts2.push_back(kp2[match.trainIdx].pt);   // 左图匹配点
    }

    // 8. 计算单应性矩阵（RANSAC抗噪，重投影误差2.0）
    Mat homo = findHomography(pts1, pts2, RANSAC, 2.0);
    if (homo.empty()) {
        cerr << "错误：单应性矩阵计算失败！" << endl;
        return false;
    }

    // 9. 计算画布尺寸（解决ROI越界核心逻辑）
    vector<Point2f> transformed_corners;
    calcCorners(homo, img1, transformed_corners);

    // 计算所有角点的极值（覆盖右图变换后+左图区域）
    float min_x = min({transformed_corners[0].x, transformed_corners[1].x, transformed_corners[2].x, transformed_corners[3].x, 0.0f});
    float min_y = min({transformed_corners[0].y, transformed_corners[1].y, transformed_corners[2].y, transformed_corners[3].y, 0.0f});
    float max_x = max({transformed_corners[0].x, transformed_corners[1].x, transformed_corners[2].x, transformed_corners[3].x, (float)img2.cols});
    float max_y = max({transformed_corners[0].y, transformed_corners[1].y, transformed_corners[2].y, transformed_corners[3].y, (float)img2.rows});

    // 画布最终尺寸
    int canvas_width = (int)ceil(max_x - min_x);
    int canvas_height = (int)ceil(max_y - min_y);

    // 偏移矩阵（处理负坐标，避免越界）
    Mat offset_mat = Mat::eye(3, 3, CV_64FC1);
    if (min_x < 0) offset_mat.at<double>(0, 2) = -min_x;
    if (min_y < 0) offset_mat.at<double>(1, 2) = -min_y;

    // 10. 透视变换（叠加偏移矩阵）
    Mat final_homo = offset_mat * homo;
    Mat warpImg1;
    warpPerspective(img1, warpImg1, final_homo, Size(canvas_width, canvas_height));

    // 11. 拼接图像（左图拷贝时校验ROI边界）
    Mat resultImg = Mat::zeros(canvas_height, canvas_width, CV_8UC3);
    warpImg1.copyTo(resultImg); // 先拷贝变换后的右图

    // 计算左图拷贝位置，确保ROI不越界
    int img2_x = (min_x < 0) ? -min_x : 0;
    int img2_y = (min_y < 0) ? -min_y : 0;
    Rect img2_roi = Rect(
        img2_x,
        img2_y,
        min(img2.cols, canvas_width - img2_x),
        min(img2.rows, canvas_height - img2_y)
    );
    img2(Rect(0, 0, img2_roi.width, img2_roi.height)).copyTo(resultImg(img2_roi));

    // ========== 截取中间区域，高度=左图高度 ==========
    int target_h = img2.rows; // 左图高度（目标高度）
    // 计算垂直中间起始位置：(拼接图高度 - 左图高度) / 2
    int start_y = (resultImg.rows - target_h) / 2.2;
    // 确保起始位置不越界（避免负数）
    start_y = max(start_y, 0);
    // 截取：宽度不变，高度=左图高度，起始y为中间位置
    Rect crop_roi(0, start_y, resultImg.cols, target_h);
    // 最终结果（只取中间区域）
    resultImg = resultImg(crop_roi).clone();
    // =====================================================
    // 12. 保存+缩放显示结果
    imwrite(savePath, resultImg);

    // 缩放显示
    Mat resized_img1 = resizeImage(img1, 800, 800);
    Mat resized_img2 = resizeImage(img2, 800, 800);
    Mat resized_result = resizeImage(resultImg, 1500, 800);

    imshow("右图（待变换）", resized_img1);
    imshow("左图（基准）", resized_img2);
    imshow("ORB算法拼接结果（已缩放）", resized_result);
    cout << "拼接完成！结果已保存至：" << savePath << endl;

     // 拼接+计时


    waitKey(0);
    destroyAllWindows();
    return true;
}

// 主
int main() {
    // ========== （绝对路径，无中文/空格） ==========(xr,cc,fangzi)
     string imgPath1 = "C:\\jjjtp\\ccy.png";  // 右图（透视变换）
     string imgPath2 = "C:\\jjjtp\\ccz.png";   // 左图（基准图）
     string savePath = "C:\\images\\stitch_result.jpg"; // 结果保存路径


    // 执行
    bool success = orbImageStitch(imgPath1, imgPath2, savePath);
    return success ? 0 : -1;
}