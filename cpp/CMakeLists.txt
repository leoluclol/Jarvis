cmake_minimum_required(VERSION 3.16)
project(jarvis_cpp VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

option(JARVIS_BUILD_TESTS "Build the C++ test suite" ON)

# ---------------------------------------------------------------------------
# External dependencies
# ---------------------------------------------------------------------------
find_package(PkgConfig REQUIRED)
pkg_check_modules(PORTAUDIO REQUIRED IMPORTED_TARGET portaudio-2.0)
pkg_check_modules(CURL REQUIRED IMPORTED_TARGET libcurl)
pkg_check_modules(ALSA IMPORTED_TARGET alsa)
find_path(NLOHMANN_JSON_INCLUDE_DIR nlohmann/json.hpp REQUIRED)

# ONNX Runtime (Silero VAD). Prebuilt releases ship include/ + lib/.
set(ONNXRUNTIME_ROOT "" CACHE PATH
    "Root of an ONNX Runtime install containing include/ and lib/")
find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h
  HINTS
    ${ONNXRUNTIME_ROOT}/include
    /usr/local/include/onnxruntime
    /usr/include/onnxruntime
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/onnxruntime/include)
find_library(ONNXRUNTIME_LIB onnxruntime
  HINTS
    ${ONNXRUNTIME_ROOT}/lib
    /usr/local/lib
    /usr/lib
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/onnxruntime/lib)
if(NOT ONNXRUNTIME_INCLUDE_DIR OR NOT ONNXRUNTIME_LIB)
  message(FATAL_ERROR
    "ONNX Runtime not found. Install it or pass -DONNXRUNTIME_ROOT=<prefix>.\n"
    "  See cpp/scripts/setup_deps.sh")
endif()

# Repo layout: cpp/ sits inside the Jarvis repo. These get baked into the
# binary so it can locate models/ and audio/ regardless of the CWD.
get_filename_component(JARVIS_REPO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/.." ABSOLUTE)
add_compile_definitions(
  JARVIS_REPO_ROOT="${JARVIS_REPO_ROOT}"
  JARVIS_CPP_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
)

# ---------------------------------------------------------------------------
# Vendored TFLite Micro audio frontend (Apache-2.0), fixed-point 16-bit.
# Same sources + flags used by the upstream pymicro_features package.
# ---------------------------------------------------------------------------
set(MICROFRONTEND_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/microfrontend)
set(MICROFRONTEND_LIB_DIR ${MICROFRONTEND_DIR}/tensorflow/lite/experimental/microfrontend/lib)

add_library(microfrontend STATIC
  ${MICROFRONTEND_DIR}/kissfft/kiss_fft.cc
  ${MICROFRONTEND_DIR}/kissfft/tools/kiss_fftr.cc
  ${MICROFRONTEND_LIB_DIR}/kiss_fft_int16.cc
  ${MICROFRONTEND_LIB_DIR}/fft.cc
  ${MICROFRONTEND_LIB_DIR}/fft_util.cc
  ${MICROFRONTEND_LIB_DIR}/filterbank.cc
  ${MICROFRONTEND_LIB_DIR}/filterbank_util.cc
  ${MICROFRONTEND_LIB_DIR}/frontend.cc
  ${MICROFRONTEND_LIB_DIR}/frontend_util.cc
  ${MICROFRONTEND_LIB_DIR}/log_lut.cc
  ${MICROFRONTEND_LIB_DIR}/log_scale.cc
  ${MICROFRONTEND_LIB_DIR}/log_scale_util.cc
  ${MICROFRONTEND_LIB_DIR}/noise_reduction.cc
  ${MICROFRONTEND_LIB_DIR}/noise_reduction_util.cc
  ${MICROFRONTEND_LIB_DIR}/pcan_gain_control.cc
  ${MICROFRONTEND_LIB_DIR}/pcan_gain_control_util.cc
  ${MICROFRONTEND_LIB_DIR}/window.cc
  ${MICROFRONTEND_LIB_DIR}/window_util.cc
)
target_compile_definitions(microfrontend PRIVATE FIXED_POINT=16)
target_include_directories(microfrontend PUBLIC
  ${MICROFRONTEND_DIR}
  ${MICROFRONTEND_DIR}/kissfft
)

# ---------------------------------------------------------------------------
# Core library
# ---------------------------------------------------------------------------
add_library(jarvis_core STATIC
  src/config.cpp
  src/env.cpp
  src/wav.cpp
  src/hardware_controller.cpp
  src/audio_controller.cpp
  src/wakeword_engine.cpp
  src/vad.cpp
  src/openai_client.cpp
  src/conversation_manager.cpp
  src/assistant_orchestrator.cpp
)
target_include_directories(jarvis_core PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${NLOHMANN_JSON_INCLUDE_DIR}
  ${ONNXRUNTIME_INCLUDE_DIR}
)
target_link_libraries(jarvis_core PUBLIC
  microfrontend
  PkgConfig::PORTAUDIO
  PkgConfig::CURL
  ${ONNXRUNTIME_LIB}
  pthread
)
if(ALSA_FOUND)
  target_compile_definitions(jarvis_core PUBLIC JARVIS_HAS_ALSA)
  target_link_libraries(jarvis_core PUBLIC PkgConfig::ALSA)
endif()
target_compile_options(jarvis_core PRIVATE -Wall -Wextra)
# The TFLite C runtime is dlopen()ed at runtime (like pymicro_wakeword), so no
# link-time dependency is needed; ensure rpath lets it find bundled libs.
set_target_properties(jarvis_core PROPERTIES
  INSTALL_RPATH "$ORIGIN/../third_party/libtensorflowlite_c"
)

# ---------------------------------------------------------------------------
# Main executable
# ---------------------------------------------------------------------------
add_executable(jarvis src/main.cpp)
target_link_libraries(jarvis PRIVATE jarvis_core)

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
if(JARVIS_BUILD_TESTS)
  enable_testing()
  add_executable(test_conversation_manager tests/test_conversation_manager.cpp)
  target_link_libraries(test_conversation_manager PRIVATE jarvis_core)
  target_compile_options(test_conversation_manager PRIVATE -Wall -Wextra)

  add_executable(test_wakeword tests/test_wakeword.cpp)
  target_link_libraries(test_wakeword PRIVATE jarvis_core)
  target_compile_options(test_wakeword PRIVATE -Wall -Wextra)
  find_package(Threads REQUIRED)

  add_executable(test_vad tests/test_vad.cpp)
  target_link_libraries(test_vad PRIVATE jarvis_core)
  target_compile_options(test_vad PRIVATE -Wall -Wextra)

  add_test(NAME conversation_manager COMMAND test_conversation_manager)
  add_test(NAME wakeword COMMAND test_wakeword)
  add_test(NAME vad COMMAND test_vad)
endif()
