#include <pylon/PylonIncludes.h>
#include <pylon/BaslerUniversalInstantCamera.h>
#include <iostream>
#include <thread>
#include <chrono>
#include <csignal>

#include "camera_config.h"
#include "ferret_tracker.h"

using namespace Pylon;

static volatile bool g_running = true;

void signal_handler(int) { g_running = false; }

int main() {
    std::signal(SIGINT,  signal_handler);
    std::signal(SIGTERM, signal_handler);

    PylonInitialize();

    try {
        CBaslerUniversalInstantCamera camera(
            CTlFactory::GetInstance().CreateFirstDevice());

        std::cout << "Camera: " << camera.GetDeviceInfo().GetModelName() << "\n";

        configure_camera(camera);

        FerretTracker tracker;
        camera.RegisterImageEventHandler(&tracker,
            RegistrationMode_Append, Cleanup_None);

        std::cout << "Warming up background model — keep arena empty for 30s...\n";
        camera.StartGrabbing(GrabStrategy_LatestImageOnly,
                             GrabLoop_ProvidedByInstantCamera);

        while (g_running) {
            std::this_thread::sleep_for(std::chrono::milliseconds(5));

            if (!tracker.ferret.valid || !tracker.prey.valid) continue;

            // Inter-animal distance
            float dx = tracker.ferret.pos_mm.x - tracker.prey.pos_mm.x;
            float dy = tracker.ferret.pos_mm.y - tracker.prey.pos_mm.y;
            float distance_mm = std::sqrt(dx * dx + dy * dy);

            std::printf(
                "Ferret: (%.0f, %.0f)mm  %.0fmm/s  %.0fdeg  |  "
                "Prey: (%.0f, %.0f)mm  %.0fmm/s  %.0fdeg  |  "
                "Dist: %.0fmm\n",
                tracker.ferret.pos_mm.x, tracker.ferret.pos_mm.y,
                tracker.ferret.speed_mm_s,
                tracker.ferret.direction_deg,
                tracker.prey.pos_mm.x, tracker.prey.pos_mm.y,
                tracker.prey.speed_mm_s,
                tracker.prey.direction_deg,
                distance_mm);
        }

        camera.StopGrabbing();
        std::cout << "Stopped.\n";

    } catch (const GenericException& e) {
        std::cerr << "Pylon error: " << e.GetDescription() << "\n";
        PylonTerminate();
        return 1;
    }

    PylonTerminate();
    return 0;
}
