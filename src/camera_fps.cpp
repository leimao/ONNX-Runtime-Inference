
#include "opencv2/opencv.hpp"
#include <opencv2/videoio.hpp>
#include <time.h>

using namespace cv;
using namespace std;

int main(int argc, char** argv)
{

    // Start default camera
    VideoCapture video(0);

    int deviceID = 0;             // 0 = open default camera
    int apiID = cv::CAP_ANY;      // 0 = autodetect default API
    // int apiID = cv::CAP_DSHOW;
    // open selected camera using selected API
    video.open(deviceID, apiID);
    if (!video.isOpened()) {
        std::cerr << "ERROR! Unable to open camera" << std::endl;
        return -1;
    }
    // With webcam get(CV_CAP_PROP_FPS) does not work.
    // Let's see for ourselves.

    // double fps = video.get(CV_CAP_PROP_FPS);
    // If you do not care about backward compatibility
    // You can use the following instead for OpenCV 3
    double fps = video.get(CAP_PROP_FPS);
    cout << "Frames per second using video.get(CAP_PROP_FPS) : " << fps << endl;

    // Number of frames to capture
    int num_frames = 120;

    // Start and end times
    time_t start, end;

    // Variable for storing video frames
    Mat frame;

    cout << "Capturing " << num_frames << " frames" << endl ;

    // Start time
    time(&start);

    // Grab a few frames
    for(int i = 0; i < num_frames; i++)
    {
        video >> frame;
    }

    // End Time
    time(&end);

    // Time elapsed
    double seconds = difftime (end, start);
    cout << "Time taken : " << seconds << " seconds" << endl;

    // Calculate frames per second
    fps  = num_frames / seconds;
    cout << "Estimated frames per second : " << fps << endl;

    // Release video
    video.release();
    return 0;
}