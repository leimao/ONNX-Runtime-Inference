
// https://github.com/microsoft/onnxruntime/blob/rel-1.6.0/csharp/test/Microsoft.ML.OnnxRuntime.EndToEndTests.Capi/CXX_Api_Sample.cpp
// https://github.com/microsoft/onnxruntime/blob/rel-1.6.0/include/onnxruntime/core/session/onnxruntime_cxx_api.h
#include <cuda_provider_factory.h>
#include <onnxruntime_cxx_api.h>

#include <opencv2/dnn/dnn.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>

#include <chrono>
#include <cmath>
#include <exception>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <string>
#include <vector>
#include <thread>

#include "commons.h"

using namespace cv;
using namespace std;

void readFrames(cv::VideoCapture& cap, FixedQueue<cv::Mat>& framesQueue)
{
    cv::Mat frame;
    while (true)
    {
        cap >> frame;
        if (!frame.empty())
        {
            framesQueue.enqueue(frame.clone());
        }
        else
        {
            break;
        }
    }
}

int main(int argc, char* argv[])
{
    const std::string windowName = "Neural Camera";

    // https://docs.opencv.org/master/d8/dfe/classcv_1_1VideoCapture.html
    cv::Mat frame;
    //--- INITIALIZE VIDEOCAPTURE
    cv::VideoCapture cap;
    // open the default camera using default API
    // cap.open(0);
    // OR advance usage: select any API backend
    int deviceID = 0;             // 0 = open default camera
    int apiID = cv::CAP_ANY;      // 0 = autodetect default API
    // open selected camera using selected API
    cap.open(deviceID, apiID);
    // check if we succeeded
    if (!cap.isOpened()) {
        std::cerr << "ERROR! Unable to open camera" << std::endl;
        return -1;
    }

    size_t framesQueueSize = 100;
    FixedQueue<cv::Mat> framesQueue{framesQueueSize};

    std::thread framesThread{readFrames, std::ref(cap), std::ref(framesQueue)};

    // Postprocessing and rendering loop
    while (cv::waitKey(1) < 0)
    {
        cv::Mat frame = framesQueue.dequeue();
        std::string label = format("Camera: %.2f FPS", framesQueue.getThroughput());
        cv::putText(frame, label, Point(0, 15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0));
        cv::imshow(windowName, frame);
    }

    framesThread.join();



    // //--- GRAB AND WRITE LOOP
    // cout << "Start grabbing" << endl
    //     << "Press any key to terminate" << endl;
    // for (;;)
    // {
    //     // wait for a new frame from camera and store it into 'frame'
    //     cap.read(frame);
    //     // check if we succeeded
    //     if (frame.empty()) {
    //         cerr << "ERROR! blank frame grabbed\n";
    //         break;
    //     }
    //     // show live and wait for a key with timeout long enough to show images
    //     cv::imshow("Live", frame);
    //     if (cv::waitKey(5) >= 0)
    //         break;
    // }
    // the camera will be deinitialized automatically in VideoCapture destructor
    return 0;
}
