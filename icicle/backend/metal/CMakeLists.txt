cmake_minimum_required(VERSION 3.18)

project(icicle_metal_backend LANGUAGES CXX)

include(cmake/shaders.cmake)


# TODO Yuval: need to strip the libs after building

# Set minimum OS X deployment target
set(CMAKE_OSX_DEPLOYMENT_TARGET "11.0" CACHE STRING "Minimum OS X deployment target" FORCE)
# TODO: handle IOS too

find_library(METAL_FRAMEWORK Metal REQUIRED)
find_library(FOUNDATION_FRAMEWORK Foundation REQUIRED)
find_library(QUARTZCORE_FRAMEWORK QuartzCore REQUIRED)

# Metal API library
add_library(metal_api STATIC src/metal_api.cpp)
target_include_directories(metal_api PRIVATE include)

# device API library
add_library(icicle_metal_device SHARED src/metal_device_api.cpp)
target_include_directories(icicle_metal_device PRIVATE include)
target_link_libraries(icicle_metal_device PUBLIC metal_api pthread)
target_link_libraries(icicle_metal_device PRIVATE icicle_device ${METAL_FRAMEWORK} ${FOUNDATION_FRAMEWORK} ${QUARTZCORE_FRAMEWORK})
set_target_properties(icicle_metal_device PROPERTIES OUTPUT_NAME "icicle_backend_metal_device")


install(TARGETS icicle_metal_device
    RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/backend/metal
    LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/backend/metal
    ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/backend/metal)

# ALlow undefined symbols when building dylib to avoid dependency on metal_device library
# Note that the symbols will be resolved at runtime and this simplifies the loading process
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -undefined dynamic_lookup")

# field API library
if (FIELD)
    string(TOLOWER "${FIELD}" FIELD_LOWERCASE)

    # The following call is compiling the shaders for the field and generates a cpp file named ${METALLIB_COMPILED_AS_CPP} to link against
    set(METALLIB_COMPILED_AS_CPP "${PROJECT_BINARY_DIR}/field_${FIELD_LOWERCASE}_metallib.cpp")
    set(SHADERS_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/shaders")
    set(SHADER_SOURCES_DIR "${SHADERS_ROOT_DIR}/fields/${FIELD_LOWERCASE}")
    set(LIB_PREFIX_NAME "field_${FIELD_LOWERCASE}")
    compile_shaders_to_cpp_source(${LIB_PREFIX_NAME} ${SHADERS_ROOT_DIR} ${SHADER_SOURCES_DIR} ${METALLIB_COMPILED_AS_CPP})
    # the target metal_field_shaders handles the shader compilation so only need to depend on it to have this cpp source in the build dir
    add_custom_target(metal_field_shaders ALL DEPENDS ${METALLIB_COMPILED_AS_CPP})

    # link the backend with the cpp file. This embeds the data in the elf and gives a symbol for the data and length.
    set(METAL_FIELD_SOURCES ${METALLIB_COMPILED_AS_CPP} src/field/metal_vec_ops.cpp)
    if (NTT)
        list(APPEND METAL_FIELD_SOURCES src/field/metal_ntt.cpp)
        list(APPEND METAL_FIELD_SOURCES src/polynomials/metal_polynomial_backend.cpp)
    endif()
    if(POSEIDON)
        list(APPEND METAL_FIELD_SOURCES src/hash/metal_poseidon.cpp)
    endif()
    if(POSEIDON2)
        list(APPEND METAL_FIELD_SOURCES src/hash/metal_poseidon2.cpp)
    endif()
    add_library(icicle_metal_field SHARED ${METAL_FIELD_SOURCES})
    set_target_properties(icicle_metal_field PROPERTIES OUTPUT_NAME "icicle_backend_metal_field_${FIELD}")
    target_include_directories(icicle_metal_field PRIVATE include)
    add_dependencies(icicle_metal_field metal_field_shaders)
    target_link_libraries(icicle_metal_field PRIVATE metal_api icicle_field ${METAL_FRAMEWORK} ${FOUNDATION_FRAMEWORK} ${QUARTZCORE_FRAMEWORK})

    # define a unique identifier for this library
    string(TOUPPER "${LIB_PREFIX_NAME}" LIB_PREFIX_UPPERCASE)
    target_compile_definitions(icicle_metal_field PRIVATE METAL_LIBRARY_ID_"${LIB_PREFIX_UPPERCASE}")

    install(TARGETS icicle_metal_field
        RUNTIME DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/backend/${FIELD}/metal"
        LIBRARY DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/backend/${FIELD}/metal"
        ARCHIVE DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/backend/${FIELD}/metal")
endif() # FIELD

# curve API library
if (CURVE)
    string(TOLOWER "${CURVE}" CURVE_LOWERCASE)

    # The following call is compiling the shaders for the field and generates a cpp file named ${METALLIB_COMPILED_AS_CPP} to link against
    set(METALLIB_COMPILED_AS_CPP "${PROJECT_BINARY_DIR}/curve_${CURVE_LOWERCASE}_metallib.cpp")
    set(SHADERS_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/shaders")
    set(SHADER_SOURCES_DIR "${SHADERS_ROOT_DIR}/curves/${CURVE_LOWERCASE}")
    set(LIB_PREFIX_NAME "curve_${CURVE_LOWERCASE}")

    compile_shaders_to_cpp_source(${LIB_PREFIX_NAME} ${SHADERS_ROOT_DIR} ${SHADER_SOURCES_DIR} ${METALLIB_COMPILED_AS_CPP})
    # the target metal_curve_shaders handles the shader compilation so only need to depend on it to have this cpp source in the build dir
    add_custom_target(metal_curve_shaders ALL DEPENDS ${METALLIB_COMPILED_AS_CPP})

    # link the backend with the cpp file. This embeds the data in the elf and gives a symbol for the data and length.
    set(METAL_CURVE_SOURCES ${METALLIB_COMPILED_AS_CPP} src/curve/metal_mont_conversion.cpp)
    if(MSM)
        list(APPEND METAL_CURVE_SOURCES src/msm/metal_msm.cpp)
    endif()
    if(G2)
        list(APPEND METAL_CURVE_SOURCES src/msm/metal_msm_g2.cpp)
    endif()
    if(ECNTT)
        list(APPEND METAL_CURVE_SOURCES src/curve/metal_ecntt.cpp)
    endif()
    add_library(icicle_metal_curve SHARED ${METAL_CURVE_SOURCES})
    set_target_properties(icicle_metal_curve PROPERTIES OUTPUT_NAME "icicle_backend_metal_curve_${CURVE}")
    target_include_directories(icicle_metal_curve PRIVATE include)
    add_dependencies(icicle_metal_curve metal_curve_shaders)
    target_link_libraries(icicle_metal_curve PRIVATE metal_api icicle_curve ${METAL_FRAMEWORK} ${FOUNDATION_FRAMEWORK} ${QUARTZCORE_FRAMEWORK})

    # define a unique identifier for this library
    string(TOUPPER "${LIB_PREFIX_NAME}" LIB_PREFIX_UPPERCASE)
    target_compile_definitions(icicle_metal_curve PRIVATE METAL_LIBRARY_ID_"${LIB_PREFIX_UPPERCASE}")

    install(TARGETS icicle_metal_curve
        RUNTIME DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/backend/${CURVE}/metal"
        LIBRARY DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/backend/${CURVE}/metal"
        ARCHIVE DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/backend/${CURVE}/metal")
endif()

# Hash library
if (HASH)
    # The following call is compiling the shaders for the field and generates a cpp file named ${METALLIB_COMPILED_AS_CPP} to link against
    set(METALLIB_COMPILED_AS_CPP "${PROJECT_BINARY_DIR}/hash_metallib.cpp")
    set(SHADERS_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/shaders")
    set(SHADER_SOURCES_DIR "${SHADERS_ROOT_DIR}/hash")
    set(LIB_PREFIX_NAME "hash")

    compile_shaders_to_cpp_source(${LIB_PREFIX_NAME} ${SHADERS_ROOT_DIR} ${SHADER_SOURCES_DIR} ${METALLIB_COMPILED_AS_CPP})
    add_custom_target(metal_hash_shaders ALL DEPENDS ${METALLIB_COMPILED_AS_CPP})

    set(METAL_HASH_SOURCES ${METALLIB_COMPILED_AS_CPP} src/hash/metal_keccak.cpp)
    list(APPEND METAL_HASH_SOURCES src/hash/metal_blake2s.cpp)
    list(APPEND METAL_HASH_SOURCES src/hash/metal_blake3.cpp)
    list(APPEND METAL_HASH_SOURCES src/hash/metal_merkle_tree.cpp)
    add_library(icicle_metal_hash SHARED ${METAL_HASH_SOURCES})
    set_target_properties(icicle_metal_hash PROPERTIES OUTPUT_NAME "icicle_backend_metal_hash")
    target_include_directories(icicle_metal_hash PRIVATE include)
    target_link_libraries(icicle_metal_hash PRIVATE metal_api icicle_hash ${METAL_FRAMEWORK} ${FOUNDATION_FRAMEWORK} ${QUARTZCORE_FRAMEWORK})

    # define a unique identifier for this library
    string(TOUPPER "${LIB_PREFIX_NAME}" LIB_PREFIX_UPPERCASE)
    target_compile_definitions(icicle_metal_hash PRIVATE METAL_LIBRARY_ID_"${LIB_PREFIX_UPPERCASE}")

    install(TARGETS icicle_metal_hash
        RUNTIME DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/backend/metal"
        LIBRARY DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/backend/metal"
        ARCHIVE DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/backend/metal")
endif()

if (BUILD_TESTS)
  add_subdirectory(tests)
endif()
