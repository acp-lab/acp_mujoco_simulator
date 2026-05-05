cmake_minimum_required(VERSION 3.8)
project(acp_mujoco_simulator)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(quadrotor_msgs REQUIRED)

if(NOT DEFINED MUJOCO_ROOT_DIR)
  if(DEFINED ENV{MUJOCO_ROOT_DIR})
    set(MUJOCO_ROOT_DIR $ENV{MUJOCO_ROOT_DIR})
  elseif(DEFINED ENV{COLCON_PAYLOAD_WS_DIR})
    set(MUJOCO_ROOT_DIR "$ENV{COLCON_PAYLOAD_WS_DIR}/mujoco-3.4.0")
  endif()
endif()

if(NOT DEFINED MUJOCO_ROOT_DIR OR NOT EXISTS ${MUJOCO_ROOT_DIR})
  message(
    FATAL_ERROR
      "Set MUJOCO_ROOT_DIR or COLCON_PAYLOAD_WS_DIR so acp_mujoco_simulator can find MuJoCo."
  )
endif()

set(MUJOCO_INCLUDE_DIR "${MUJOCO_ROOT_DIR}/include")
set(MUJOCO_LIB_DIR "${MUJOCO_ROOT_DIR}/lib")
set(MUJOCO_BIN_DIR "${MUJOCO_ROOT_DIR}/bin")

file(GLOB LIB_MUJOCO
  ${MUJOCO_BIN_DIR}/libmujoco[0-9][0-9][0-9].so
  ${MUJOCO_LIB_DIR}/libmujoco.so.*)

if(NOT LIB_MUJOCO)
  message(FATAL_ERROR "MuJoCo library not found under ${MUJOCO_ROOT_DIR}.")
endif()

add_executable(mujoco_headless src/mujoco_headless.cpp)
target_include_directories(mujoco_headless PRIVATE ${MUJOCO_INCLUDE_DIR})
target_link_libraries(mujoco_headless ${LIB_MUJOCO} dl)
ament_target_dependencies(mujoco_headless rclcpp)

install(TARGETS mujoco_headless
  DESTINATION lib/${PROJECT_NAME})

install(DIRECTORY model
DESTINATION share/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
