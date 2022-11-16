project(AISDK)
cmake_minimum_required(VERSION 3.16.3)

# include path
set(INCLUDE_PATH ${PROJECT_SOURCE_DIR}/include/)
file(GLOB_RECURSE SOURCES     
${PROJECT_SOURCE_DIR}/src/*.cpp
)

set(CMAKE_CXX_STANDARD 11)


# Compile APISDK and Link

add_compile_options(-ggdb)
add_library(aisdk ${SOURCES})

add_library(onednn SHARED IMPORTED)
set_target_properties(onednn PROPERTIES   IMPORTED_LOCATION "/usr/local/lib/libdnnl.so.2")


target_link_libraries(aisdk dnnl onednn)

# Executable Test
add_executable(main main.cpp)

target_link_libraries(main aisdk dnnl onednn)

target_include_directories(aisdk PUBLIC 
    ${PROJECT_SOURCE_DIR}/include/data/
    ${PROJECT_SOURCE_DIR}/include/env/
    ${PROJECT_SOURCE_DIR}/include/handy/
    ${PROJECT_SOURCE_DIR}/include/model/
)
target_include_directories(main PUBLIC 
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/include/data/
    ${PROJECT_SOURCE_DIR}/include/env/
    ${PROJECT_SOURCE_DIR}/include/handy/
    ${PROJECT_SOURCE_DIR}/include/model/
)

