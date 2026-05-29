#pragma once

#include <pylon/PylonIncludes.h>
#include <pylon/BaslerUniversalInstantCamera.h>
#include <opencv2/opencv.hpp>
#include "tracker.h"

// Camera / optics constants for a2A1920-160umPRO at 1.2m with 4mm lens
static constexpr float GSD_MM_PX = 1.035f;  // mm per pixel
static constexpr float FPS       = 200.0f;
static constexpr int   WARMUP_FRAMES = static_cast<int>(FPS * 30); // 30s BG warmup

class FerretTracker : public Pylon::CImageEventHandler {
public:
    TrackState ferret;
    TrackState prey;

    FerretTracker();

    void OnImageGrabbed(Pylon::CInstantCamera& camera,
                        const Pylon::CGrabResultPtr& result) override;

private:
    cv::Ptr<cv::BackgroundSubtractorMOG2> bg_;
    cv::KalmanFilter kf_ferret_;
    cv::KalmanFilter kf_prey_;
    cv::Mat morph_kernel_;
    uint64_t frame_count_ = 0;

    void update_track(cv::KalmanFilter& kf,
                      const std::vector<cv::Point>& contour,
                      TrackState& state);
};
