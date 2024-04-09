#include <LightStage/LightStage.hh>
#include <SpinnakerCamera/CaptureConfig.h>
#include <SpinnakerCamera/Pipeline.h>
#include <SpinnakerCamera/sinks/ExrWriter.h>
#include <cstdint>
#include <ctime>
#include <iostream>
#include <ostream>
#include <spdlog/spdlog.h>
#include <string>
#include <turntable/Turntable.h>
#include <unistd.h>

using namespace SpinCam;

int main() {
  unsigned int microsecond = 1000000;

  size_t num_leds =
      153; // 195 LEDS, 0-55 is main Arc, 56-127 cam left, 128- 195 cam right,
           // 90-153 cam back, without main arc: 56-297
  float acceleration = 1.0f;

  light_stage::LightStage ls;
  ls.init();
  ls.clearAll();
  ls.upload();

  Pipeline pipeline("/graphics/scratch2/students/siemssen/BA_Specular_Scanning/camera_acquisition/pipeline.json");
                                  
  Turntable turntable("/dev/ttyUSB0");
  turntable.setAcceleration(acceleration);

 // ls.arc->init();
  for (size_t i = 90; i <= num_leds; i++) {
    // 15 is white
    ls.setLedOnBoard(i, 15, 255);
  }
  ls.upload();

  for (uint32_t ang_count = 0; ang_count <= 1440; ang_count = ang_count + 1){
    float angle = float(ang_count) * 0.25f;

    if (turntable.isReady()) {
        turntable.moveTo(angle);
        usleep(10 * microsecond); // Wait for shake to stop
    } else {
        spdlog::error("Turntable not found");
    }

    CaptureConfig config(10000, 0);
    config.SetExposure(100000);
    config.SetFileName(fmt::format("camera_test_1_{}", ang_count));
    auto worker = pipeline.run({"Writer"}, config);
  }


  
  // Capture 
  /*
  for (uint32_t arc = 0; arc <= 25000; arc = arc + 5000) {
    ls.arc->moveTo(arc);
    usleep(10 * microsecond); // Wait for shake to stop

    for (int i = 0; i < 20; i++) {
      float ang = float(i) * 18.0f;

      if (turntable.isReady()) {
        turntable.moveTo(ang);
      } else {
        spdlog::error("Turntable not found");
      }

      int converted_arc = int(static_cast<float>(arc) / 312.5f);

      for (uint32_t exposure = 100; exposure <= 100000;
           exposure = exposure * 10) {
        {
          CaptureConfig config(10000, 0); // exposure in microseconds, gain in db
          config.SetExposure(exposure);
          // config.SetFileName("y" + std::to_string(converted_arc) + "_z" +
          //                    std::to_string(int(ang)) + "_e" +
          //                    std::to_string(exposure / 100));
          config.SetFileName(fmt::format("suuuper_langer_file_name_mit_mehr_als_15_zeichen_y{}_z{}_e{}", converted_arc, int(ang), exposure / 100));
          auto worker = pipeline.run({"Writer"}, config);
        }
      }
    }
  } */
  // Reset everything
  for (size_t i = 0; i <= num_leds; i++) {
    ls.setLedOnBoard(i, 15, 0);
  }

  if (turntable.isReady()) {
    turntable.moveTo(0);
  } else {
    spdlog::error("Turntable not found");
  }
  ls.upload();
  ls.arc->moveTo(0);
  ls.clearAll();
  ls.upload();
}
