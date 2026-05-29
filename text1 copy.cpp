#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/features2d/features2d.hpp>

using namespace cv;
using namespace std;

int main() {


    // 初始化视频参数
    cv::VideoCapture cap1("C:\\Users\\27722\\Desktop\\jjjtp\\gou1.mp4");
    cv::VideoCapture cap2("C:\\Users\\27722\\Desktop\\jjjtp\\gou2.mp4");

 while (1)
    {
       
    // 读取第一个视频的第一帧,采集视频帧
    cv::Mat captureImage1;Mat captureImage_gray1;
    cap1 >> captureImage1;
    // 读取第二个视频的第一帧，采集视频帧
    cv::Mat captureImage2;Mat captureImage_gray2;
    cap2 >> captureImage2;
    if (captureImage1.empty())//采集为空的处理
            continue;
    if (captureImage2.empty())//采集为空的处理
            continue;
    double time0 = static_cast<double>(getTickCount());//记录起始时间
    cvtColor(captureImage1, captureImage_gray1, COLOR_BGR2GRAY);//采集的视频帧转化为灰度图
    cvtColor(captureImage2, captureImage_gray2, COLOR_BGR2GRAY);//采集的视频帧转化为灰度图


    //【7】检测SIFT关键点并提取测试图像中的描述符
    vector<KeyPoint> captureKeyPoints1; vector<KeyPoint> captureKeyPoints2;
    Mat captureDescription1;Mat captureDescription2;
    Ptr<ORB> featureDetector = ORB::create();
    //【8】调用detect函数检测出特征关键点，保存在vector容器中
    featureDetector->detect(captureImage_gray1, captureKeyPoints1);
    featureDetector->detect(captureImage_gray2, captureKeyPoints2);
    //【9】计算描述符
    featureDetector->compute(captureImage_gray1, captureKeyPoints1, captureDescription1);
    featureDetector->compute(captureImage_gray2, captureKeyPoints2, captureDescription2);
     //【5】基于FLANN的描述符对象匹配
    flann::Index flannIndex(captureDescription1, flann::LshIndexParams(12, 20, 2), cvflann::FLANN_DIST_HAMMING);

    //【10】匹配和测试描述符，获取两个最邻近的描述符
    Mat matchIndex(captureDescription2.rows, 2, CV_32SC1), matchDistance(captureDescription2.rows, 2, CV_32FC1);
    flannIndex.knnSearch(captureDescription2, matchIndex, matchDistance, 2, flann::SearchParams());//调用K邻近算法

    //【11】根据劳氏算法（Lowe's algorithm）选出优秀的匹配
    vector<DMatch> goodMatches;
      for (int i = 0; i < matchDistance.rows; i++)
        {
            if (matchDistance.at<float>(i, 0) < 0.6 * matchDistance.at<float>(i, 1))
            {
                DMatch dmatches(i, matchIndex.at<int>(i, 0), matchDistance.at<float>(i, 0));
                goodMatches.push_back(dmatches);
            }
        }        
          //【12】绘制并显示匹配窗口
        Mat resultImage;
        drawMatches(captureImage2, captureKeyPoints2, captureImage1, captureKeyPoints1,  goodMatches, resultImage);
        imshow("匹配窗口", resultImage);

        //【13】显示帧率
        cout << ">帧率= " << getTickFrequency() / (getTickCount() - time0) << endl;

        //按下ESC键，则程序退出
        if (char(waitKey(1)) == 27) break;
    }

    return 0;
}