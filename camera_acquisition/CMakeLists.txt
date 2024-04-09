cmake_minimum_required(VERSION 3.15)

project(
  test_lightstage_cmake
  DESCRIPTION "Test project to see if cmake installation of lightstage libraries works"
  VERSION 1.0.0
)

########## This is completely optional. Just stuff I like to use. #######
include(cmake/StandardProjectSettings.cmake)

### Add some more project options by linking this library
add_library(project_options INTERFACE)
# cpp20 standard
target_compile_features(project_options INTERFACE cxx_std_20)

## Some optional settings that can be enabled in ccmake
# add sanitizer options
include(cmake/Sanitizer.cmake)
enable_sanitizers(project_options)

# add doxygen option
include(cmake/Doxygen.cmake)
enable_doxygen()

# add static analysis with clang-tidy as option
include(cmake/StaticAnalyzers.cmake)

### Tell the compiler to complain about warnings
# by linking this library
add_library(project_warnings INTERFACE)
include(cmake/CompilerWarnings.cmake)
set_project_warnings(project_warnings)


######### This is the important part now: #####################

set(
  JSON_ISNTALL_DIR
  "/graphics/opt/opt_Ubuntu22.04/json/"
  CACHE PATH "Installation directory of nlohmann/json"
)
find_package(nlohmann_json REQUIRED PATHS "${JSON_ISNTALL_DIR}")

# After compiling and installing the libraries from code_public
# you can tell cmake about them linke this:
find_package(LightStage REQUIRED)
find_package(motor_arc REQUIRED)
find_package(turntable REQUIRED)
find_package(USBComm REQUIRED)
find_package(SpinnakerCamera REQUIRED)
find_package(fmt REQUIRED)
find_package(OpenEXR REQUIRED)
find_package(
  Qt5
  COMPONENTS Widgets Core Gui CONFIG
  REQUIRED
)
find_package(OpenMP)
# This is the program we want to compile from the source file(s)
add_executable(test test.cpp)

# Tell cmake about all the dependencies needed to build this executable
target_link_libraries(
  test PUBLIC 
  LightStage::LightStage
  turntable::turntable
  nlohmann_json::nlohmann_json
  # fmt::fmt
  # OpenEXR::IlmImf
  SpinnakerCamera::SpinnakerCamera
  OpenMP::OpenMP_CXX
  # Qt5::Core Qt5::Widgets Qt5::Gui
  USBComm::USBComm motor_arc::Arc
  project_options  # (optional) use the project options defined above
  project_warnings # (optional) tell the compiler to complain about all the things in cmake/CompilerWarnings.cmake
)
