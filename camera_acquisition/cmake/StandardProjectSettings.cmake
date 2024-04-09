# Set default build type if none was specified
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to 'Release' as none was specified.")
  set(CMAKE_BUILD_TYPE
      Release
      CACHE STRING "Choose the type of build." FORCE)
  # Set possible values for build type
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release")
endif(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)

# Set install path based on build type
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    if (CMAKE_BUILD_TYPE MATCHES "Release")
      set(CMAKE_INSTALL_PREFIX
          "${CMAKE_SOURCE_DIR}/Release"
          CACHE PATH "default install path" FORCE)
  elseif (CMAKE_BUILD_TYPE MATCHES "Debug")
      set(CMAKE_INSTALL_PREFIX
          "${CMAKE_SOURCE_DIR}/Debug"
          CACHE PATH "default install path" FORCE)
  endif()
endif()

# Use ccache if availiable
find_program(CCACHE ccache)
if(CCACHE)
  message("using ccache")
  set(CMAKE_CXX_COMPILER_LAUNCHER ${CCACHE})
else()
  message("ccache not found - cannot use")
endif(CCACHE)

# Generate compile_commands.json
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Add option for interprocedural optimization, aka link time optimiztations
# (LTO)
option(ENABLE_IPO
       "Enable interprocedural Optimization, aka Link Time Optimiation (LTO)"
       OFF)
if(ENABLE_IPO)
  include(CheckIPOSupported)
  check_ipo_supported(RESULT result OUTPUT output)
  if(result)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
  else()
    message(SEND_ERROR "IPO is not supported: ${output}")
  endif()
endif()
