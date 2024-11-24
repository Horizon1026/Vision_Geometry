aux_source_directory( . AUX_SRC_VISION_GEOMETRY_LINE_TRIANGULATOR )

# Add all relative components of slam utility.
set( SLAM_UTILITY_PATH ${PROJECT_SOURCE_DIR}/../Slam_Utility )
if ( NOT TARGET lib_slam_utility_basic_type )
    add_subdirectory( ${SLAM_UTILITY_PATH}/src/basic_type ${PROJECT_SOURCE_DIR}/build/lib_slam_utility_basic_type )
endif()
if ( NOT TARGET lib_slam_utility_math )
    add_subdirectory( ${SLAM_UTILITY_PATH}/src/math ${PROJECT_SOURCE_DIR}/build/lib_slam_utility_math )
endif()
if ( NOT TARGET lib_slam_utility_operate )
    add_subdirectory( ${SLAM_UTILITY_PATH}/src/operate ${PROJECT_SOURCE_DIR}/build/lib_slam_utility_operate )
endif()
if ( NOT TARGET lib_slam_utility_log )
    add_subdirectory( ${SLAM_UTILITY_PATH}/src/log ${PROJECT_SOURCE_DIR}/build/lib_slam_utility_log )
endif()

# Add all relative components of slam utility data type.
if ( NOT TARGET lib_line_segment )
    add_subdirectory( ${SLAM_UTILITY_PATH}/src/data_type/line_segment ${PROJECT_SOURCE_DIR}/build/lib_line_segment )
endif()
if ( NOT TARGET lib_plane )
    add_subdirectory( ${SLAM_UTILITY_PATH}/src/data_type/plane ${PROJECT_SOURCE_DIR}/build/lib_plane )
endif()

add_library( lib_vision_geometry_line_triangulator ${AUX_SRC_VISION_GEOMETRY_LINE_TRIANGULATOR} )
target_include_directories( lib_vision_geometry_line_triangulator PUBLIC . )
target_link_libraries( lib_vision_geometry_line_triangulator
    lib_slam_utility_basic_type
    lib_slam_utility_math
    lib_slam_utility_operate
    lib_slam_utility_log

    lib_line_segment
    lib_plane
)
