aux_source_directory( . AUX_SRC_VISION_GEOMETRY )

if ( NOT TARGET lib_slam_utility_datatype )
    add_subdirectory( datatype ${PROJECT_SOURCE_DIR}/build/lib_slam_utility_datatype )
endif()

if ( NOT TARGET lib_slam_utility_math )
    add_subdirectory( math ${PROJECT_SOURCE_DIR}/build/lib_slam_utility_math )
endif()

if ( NOT TARGET lib_slam_utility_operate )
    add_subdirectory( operate ${PROJECT_SOURCE_DIR}/build/lib_slam_utility_operate )
endif()

add_library( lib_vision_geometry ${AUX_SRC_VISION_GEOMETRY} )
target_include_directories( lib_vision_geometry PUBLIC . )
target_link_libraries( lib_vision_geometry
    lib_slam_utility_datatype
    lib_slam_utility_math
    lib_slam_utility_operate
)
