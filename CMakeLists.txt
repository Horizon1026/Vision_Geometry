# cmake工程最低版本为3.1
cmake_minimum_required( VERSION 3.10 )

# cmake工程名
project( perspectiveNPoint )

# 设置编译类型为发布
set(CMAKE_BUILD_TYPE "Release")

# 添加c++ 11标准的支持
set(CMAKE_CXX_FLAGS "-std=c++11 -O2")

# 寻找OpenCV库并添加其路径
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

# 寻找Sophus的库并添加Sophus的路径
find_package( Sophus REQUIRED )
include_directories( ${Sophus_INCLUDE_DIRS} )

# 寻找Eigen库并添加其路径
include_directories( ${PROJECT_SOURCE_DIR}/include )
include_directories("/usr/include/eigen3")

# 添加可执行文件
add_executable( perspectiveNPoint
    src/main.cpp
    include/perspectiveNPoint.cpp
)

# 链接库到可执行文件
target_link_libraries( perspectiveNPoint
    ${OpenCV_LIBS}
    Sophus::Sophus  # 增加Sophus的库（否则找不到fmt/core.h文件）
)
