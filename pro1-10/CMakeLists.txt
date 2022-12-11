project(prob1-10)
aux_source_directory(./ sources)
add_executable(${PROJECT_NAME} ${sources})
target_include_directories(prob1-10 PUBLIC ./)
target_link_libraries(${PROJECT_NAME} PUBLIC ${OpenCV_LIBRARIES}
                      ${GFLAGS_LIBRARIES} 
                      ${GLOG_LIBRARIES} 
                      Threads::Threads)