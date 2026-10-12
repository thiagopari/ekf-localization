cmake_minimum_required(VERSION 3.8)
project(ekf_localization_cpp)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release)
endif()
add_compile_options(-Wall -Wextra -Wpedantic)

find_package(ament_cmake REQUIRED)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(visualization_msgs REQUIRED)

# ROS-free filter core: Eigen only.
add_library(ekf_filters src/ekf.cpp src/ukf.cpp)
target_compile_features(ekf_filters PUBLIC cxx_std_17)
target_include_directories(ekf_filters PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_link_libraries(ekf_filters PUBLIC Eigen3::Eigen)

add_executable(ekf_node src/ekf_node.cpp)
target_link_libraries(ekf_node ekf_filters rclcpp::rclcpp tf2_ros::tf2_ros
  ${geometry_msgs_TARGETS} ${nav_msgs_TARGETS} ${sensor_msgs_TARGETS}
  ${std_msgs_TARGETS} ${visualization_msgs_TARGETS})

install(DIRECTORY include/ DESTINATION include)
install(TARGETS ekf_filters EXPORT export_${PROJECT_NAME} ARCHIVE DESTINATION lib LIBRARY DESTINATION lib)
install(TARGETS ekf_node DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY launch DESTINATION share/${PROJECT_NAME})
ament_export_targets(export_${PROJECT_NAME})
ament_export_dependencies(Eigen3)

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)

  ament_add_gtest(test_filters test/test_filters.cpp)
  target_link_libraries(test_filters ekf_filters)

  # Golden equivalence: the ORIGINAL ekf.py / ukf.py write the CSV, then C++ replays it.
  set(GOLDEN_CSV ${CMAKE_CURRENT_BINARY_DIR}/golden.csv)
  add_test(NAME make_golden
    COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/test/make_golden.py ${GOLDEN_CSV})
  set_tests_properties(make_golden PROPERTIES FIXTURES_SETUP golden)
  ament_add_gtest(test_golden test/test_golden.cpp)
  target_link_libraries(test_golden ekf_filters)
  target_compile_definitions(test_golden PRIVATE GOLDEN_CSV="${GOLDEN_CSV}")
  set_tests_properties(test_golden PROPERTIES FIXTURES_REQUIRED golden)

  add_executable(bench_filters test/bench_filters.cpp)
  target_link_libraries(bench_filters ekf_filters)
endif()

ament_package()
