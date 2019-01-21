#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;
using namespace std;

void imhist(Mat bgr_image, vector<vector<int>>& bgr_histogram)
{
    
    
    for (int i = 0; i < 256; i++)
    {
        bgr_histogram[0][i] = 0;
        bgr_histogram[1][i] = 0;
        bgr_histogram[2][i] = 0;
    }
    
    
    for (int y = 0; y < bgr_image.rows; y++)
    {
        for (int x = 0; x < bgr_image.cols; x++)
        {
            bgr_histogram[0][(int)bgr_image.at<cv::Vec3b>(y, x)[0]]++;
            bgr_histogram[1][(int)bgr_image.at<cv::Vec3b>(y, x)[1]]++;
            bgr_histogram[2][(int)bgr_image.at<cv::Vec3b>(y, x)[2]]++;
        }
    }
}

void cumhist(vector<vector<int>> histogram, vector<vector<int>>& cumhistogram)
{
    cumhistogram[0][0] = histogram[0][0];
    cumhistogram[1][0] = histogram[1][0];
    cumhistogram[2][0] = histogram[2][0];
    
    for (int i = 1; i < 256; i++)
    {
        cumhistogram[0][i] = histogram[0][i] + cumhistogram[0][i - 1];
        cumhistogram[1][i] = histogram[1][i] + cumhistogram[1][i - 1];
        cumhistogram[2][i] = histogram[2][i] + cumhistogram[2][i - 1];
    }
}


void cumgoshist(vector<vector<float>> histogram, vector<vector<float>>& cumhistogram)
{
    cumhistogram[0][0] = histogram[0][0];
    cumhistogram[1][0] = histogram[1][0];
    cumhistogram[2][0] = histogram[2][0];
    
    for (int i = 1; i < 256; i++)
    {
        cumhistogram[0][i] = histogram[0][i] + cumhistogram[0][i - 1];
        cumhistogram[1][i] = histogram[1][i] + cumhistogram[1][i - 1];
        cumhistogram[2][i] = histogram[2][i] + cumhistogram[2][i - 1];
    }
}

void histDisplay(vector<vector<int>> histogram, const char* name)
{
    vector<vector<int>> hist(3);
    hist[0].resize(256);
    hist[1].resize(256);
    hist[2].resize(256);
    for (int i = 0; i < 256; i++)
    {
        hist[0][i] = histogram[0][i];
        hist[1][i] = histogram[1][i];
        hist[2][i] = histogram[2][i];
    }
    
    int hist_w = 512; int hist_h = 400;
    int bin_w = cvRound((double)hist_w / 256);
    
    Mat histImage(hist_h, hist_w, CV_8UC3, Scalar(255, 255, 255));
    
    
    float b_max = hist[0][0];
    float g_max = hist[1][0];
    float r_max = hist[2][0];
    for (int i = 1; i < 256; i++) {
        if (b_max < hist[0][i]) {
            b_max = hist[0][i];
        }
        if (g_max < hist[1][i]) {
            g_max = hist[1][i];
        }
        if (r_max < hist[2][i]) {
            r_max = hist[2][i];
        }
    }
    
    for (int i = 0; i < 256; i++) {
        hist[0][i] = ((double)hist[0][i] / b_max)*histImage.rows;
        hist[1][i] = ((double)hist[1][i] / g_max)*histImage.rows;
        hist[2][i] = ((double)hist[2][i] / r_max)*histImage.rows;
    }
    
    
    
    for (int i = 0; i < 256; i++)
    {
        line(histImage, Point(bin_w*(i), hist_h),
             Point(bin_w*(i), hist_h - hist[0][i]),
             Scalar(255, 0, 0), 1, 8, 0);
        line(histImage, Point(bin_w*(i), hist_h),
             Point(bin_w*(i), hist_h - hist[1][i]),
             Scalar(0, 255, 0), 1, 8, 0);
        line(histImage, Point(bin_w*(i), hist_h),
             Point(bin_w*(i), hist_h - hist[2][i]),
             Scalar(0, 0, 255), 1, 8, 0);
    }
    
    namedWindow(name, CV_WINDOW_AUTOSIZE);
    imshow(name, histImage);
}

void histDis(vector<vector<float>> histogram, const char* name)
{
    vector<vector<int>> hist(3);
    hist[0].resize(256);
    hist[1].resize(256);
    hist[2].resize(256);
    for (int i = 0; i < 256; i++)
    {
        hist[0][i] = histogram[0][i];
        hist[1][i] = histogram[1][i];
        hist[2][i] = histogram[2][i];
    }
    
    int hist_w = 512; int hist_h = 400;
    int bin_w = cvRound((double)hist_w / 256);
    
    Mat histImage(hist_h, hist_w, CV_8UC3, Scalar(255, 255, 255));
    
    
    float b_max = hist[0][0];
    float g_max = hist[1][0];
    float r_max = hist[2][0];
    for (int i = 1; i < 256; i++) {
        if (b_max < hist[0][i]) {
            b_max = hist[0][i];
        }
        if (g_max < hist[1][i]) {
            g_max = hist[1][i];
        }
        if (r_max < hist[2][i]) {
            r_max = hist[2][i];
        }
    }
    
    
    
    for (int i = 0; i < 256; i++) {
        hist[0][i] = ((double)hist[0][i] / b_max)*histImage.rows;
        hist[1][i] = ((double)hist[1][i] / g_max)*histImage.rows;
        hist[2][i] = ((double)hist[2][i] / r_max)*histImage.rows;
    }
    
    
    
    for (int i = 0; i < 256; i++)
    {
        line(histImage, Point(bin_w*(i), hist_h),
             Point(bin_w*(i), hist_h - hist[0][i]),
             Scalar(255, 0, 0), 1, 8, 0);
        line(histImage, Point(bin_w*(i), hist_h),
             Point(bin_w*(i), hist_h - hist[1][i]),
             Scalar(0, 255, 0), 1, 8, 0);
        line(histImage, Point(bin_w*(i), hist_h),
             Point(bin_w*(i), hist_h - hist[2][i]),
             Scalar(0, 0, 255), 1, 8, 0);
    }
    
    
    namedWindow(name, CV_WINDOW_AUTOSIZE);
    imshow(name, histImage);
}

int main()
{
    
    Mat image = imread("../../lenna.png", 1);
    
    vector<vector<int>> histogram;
    histogram.resize(3);
    histogram[0].resize(256);
    histogram[1].resize(256);
    histogram[2].resize(256);
    
    imhist(image, histogram);
    
    
    int size = image.rows * image.cols;
    float alpha = 255.0 / size;
    
    
    vector<vector<float>> PrRk(3);
    PrRk[0].resize(256);
    PrRk[1].resize(256);
    PrRk[2].resize(256);
    for (int i = 0; i < 256; i++)
    {
        PrRk[0][i] = (double)histogram[0][i] / size;
        PrRk[1][i] = (double)histogram[1][i] / size;
        PrRk[2][i] = (double)histogram[2][i] / size;
    }
    
    
    vector<vector<int>> cumhistogram(3);
    cumhistogram[0].resize(256);
    cumhistogram[1].resize(256);
    cumhistogram[2].resize(256);
    vector<vector<float>> cumgos(3);
    cumgos[0].resize(256);
    cumgos[1].resize(256);
    cumgos[2].resize(256);
    cumhist(histogram, cumhistogram);
    
    
    vector<vector<int>> Sk(3);
    Sk[0].resize(256);  //b
    Sk[1].resize(256);  //g
    Sk[2].resize(256);  //r
    for (int i = 0; i < 256; i++)
    {
        Sk[0][i] = cvRound((double)cumhistogram[0][i] * alpha);
        Sk[1][i] = cvRound((double)cumhistogram[1][i] * alpha);
        Sk[2][i] = cvRound((double)cumhistogram[2][i] * alpha);
    }
    
    
    vector<vector<float>> gos(3);
    gos[0].resize(256);
    gos[1].resize(256);
    gos[2].resize(256);
    float sigma;
    int median;
    cout << "Enter value of sigma : " << endl;
    cin >> sigma;
    cout << "Enter value of median : " << endl;
    cin >> median;
    
    
    for (int i = -median; i < 255 - median ; i++)
    {
        float value = (1 / sqrt(2 * 3.1416)*sigma)*exp(-(pow(i, 2) / (2 * pow(sigma, 2))));
        gos[0][i + median] = value;
        gos[1][i + median] = value;
        gos[2][i + median] = value;
    }
    
    histDis(gos, "Gaussian Histogram");
    cumgoshist(gos, cumgos);
    vector<vector<float>> Gz(3);
    Gz[0].resize(256);
    Gz[1].resize(256);
    Gz[2].resize(256);
    for (int i = 0; i < 256; i++)
    {
        Gz[0][i] = cvRound((double)cumgos[0][i] * alpha);
        Gz[1][i] = cvRound((double)cumgos[1][i] * alpha);
        Gz[2][i] = cvRound((double)cumgos[2][i] * alpha);
    }
    
    Mat new_image = image.clone();
    
    for (int y = 0; y < 256; y++)
    {
        for (int x = 0; x < 256; x++)
        {
            if (Sk[0][y] == Gz[0][x] || abs(Sk[0][y] - Gz[0][x]) == 1)
            {
                Sk[0][y] = x;
                break;
            }
            if (Sk[1][y] == Gz[1][x] || abs(Sk[1][y] - Gz[1][x]) == 1)
            {
                Sk[1][y] = x;
                break;
            }
            if (Sk[2][y] == Gz[2][x] || abs(Sk[2][y] - Gz[2][x]) == 1)
            {
                Sk[2][y] = x;
                break;
            }
        }
    }
    
    for (int y = 0; y < image.rows; y++)
    {
        for (int x = 0; x < image.cols; x++)
        {
            new_image.at<Vec3b>(y, x)[0] = saturate_cast<uchar>(Sk[0][image.at<Vec3b>(y, x)[0]]);
            new_image.at<Vec3b>(y, x)[1] = saturate_cast<uchar>(Sk[1][image.at<Vec3b>(y, x)[1]]);
            new_image.at<Vec3b>(y, x)[2] = saturate_cast<uchar>(Sk[2][image.at<Vec3b>(y, x)[2]]);
        }
    }
    
    
    vector<vector<float>> PsSk(3);
    PsSk[0].resize(256);
    PsSk[1].resize(256);
    PsSk[2].resize(256);
    for (int i = 0; i < 256; i++)
    {
        PsSk[0][i] = 0;
        PsSk[1][i] = 0;
        PsSk[2][i] = 0;
    }
    
    for (int i = 0; i < 256; i++)
    {
        PsSk[0][Sk[0][i]] += PrRk[0][i];
        PsSk[1][Sk[1][i]] += PrRk[1][i];
        PsSk[2][Sk[2][i]] += PrRk[2][i];
    }
    
    vector<vector<int>> final(3);
    final[0].resize(256);
    final[1].resize(256);
    final[2].resize(256);
    for (int i = 0; i < 256; i++)
    {
        final[0][i] = cvRound(PsSk[0][i] * 255);
        final[1][i] = cvRound(PsSk[1][i] * 255);
        final[2][i] = cvRound(PsSk[2][i] * 255);
    }
    
    
    namedWindow("Original Image");
    imshow("Original Image", image);
    
    histDisplay(histogram, "Original Histogram");
    
    
    
    namedWindow("Equilized Image");
    imshow("Equilized Image", new_image);
    
    histDisplay(final, "Equilized Histogram");
    
    waitKey();
    return 0;
}
