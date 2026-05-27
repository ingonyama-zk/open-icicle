cmake_minimum_required(VERSION 3.18)

if (CURVE)
    string(TOLOWER "${CURVE}" CURVE_LOWERCASE)

    set(METALLIB_COMPILED_AS_CPP "${PROJECT_BINARY_DIR}/test_metallib.cpp")
    set(SHADERS_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../shaders")
    set(SHADER_SOURCES_DIR "ec")
    set(LIB_PREFIX_NAME "test")
    compile_shaders_to_cpp_source(${LIB_PREFIX_NAME} ${SHADERS_ROOT_DIR} ${SHADER_SOURCES_DIR} ${METALLIB_COMPILED_AS_CPP})
    add_custom_target(metal_test_shaders ALL DEPENDS ${METALLIB_COMPILED_AS_CPP})

    # link the backend with the cpp file. This embeds the data in the elf and gives a symbol for the data and length.
    set(METAL_TEST_SOURCES ${METALLIB_COMPILED_AS_CPP} test_base.cpp)

    include(GoogleTest)
    include(FetchContent)
    FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/refs/tags/v1.13.0.zip
    )
        
    # Disable installation of gtest and gmock
    set(INSTALL_GTEST OFF)
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    
    # For Windows: Prevent overriding the parent project's compiler/linker settings
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
    
    # Ensure gtest and gmock are not built by default
    set_target_properties(gtest PROPERTIES EXCLUDE_FROM_ALL TRUE)
    set_target_properties(gtest_main PROPERTIES EXCLUDE_FROM_ALL TRUE)
    set_target_properties(gmock PROPERTIES EXCLUDE_FROM_ALL TRUE)
    set_target_properties(gmock_main PROPERTIES EXCLUDE_FROM_ALL TRUE)
        
    enable_testing()

    add_compile_definitions(BACKEND_BUILD_DIR="${CMAKE_BINARY_DIR}/backend" METAL_LIBRARY_ID_TEST=1)
    add_executable(test_metal_base ${METAL_TEST_SOURCES})
    target_include_directories(test_metal_base PRIVATE ../include ../../../include)
    add_dependencies(test_metal_base metal_test_shaders)
    target_link_libraries(test_metal_base PRIVATE GTest::gtest_main icicle_metal_curve icicle_metal_field icicle_metal_device icicle_device metal_api ${METAL_FRAMEWORK} ${FOUNDATION_FRAMEWORK} ${QUARTZCORE_FRAMEWORK})

    gtest_discover_tests(test_metal_base)
endif()