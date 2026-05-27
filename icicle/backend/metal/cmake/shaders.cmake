
# SOURCES_DIR is the directory containing all the shaders to compile
function(compile_shaders_to_cpp_source LIB_PREFIX_NAME SHADER_ROOT_DIR SOURCES_DIR OUTPUT_FILE)
    # Prepare include flags
    set(INCLUDE_DIRS "${SHADER_ROOT_DIR}/include" "${SHADER_ROOT_DIR}/../include")
    set(COMMAND_INCLUDE_DIRS "")
    foreach(dir ${INCLUDE_DIRS})
        list(APPEND COMMAND_INCLUDE_DIRS "-I" "${dir}")
    endforeach()

    # Glob all .metal files recursively for dependencies. Note that this means it's recompiled for any update to any shader
    file(GLOB_RECURSE METAL_DEPS "${SHADER_ROOT_DIR}/include/*")
    # We compile all shaders file for this field
    file(GLOB_RECURSE SOURCE_SHADERS "${SOURCES_DIR}/*.metal")
    message(STATUS "Compiling shaders for library ${LIB_PREFIX_NAME}, sources in ${SOURCES_DIR} to ${OUTPUT_FILE}")

    # Intermediate directory for AIR files
    set(AIR_FILES "")
    foreach(SHADER ${SOURCE_SHADERS})
        # Get the base name for the AIR file
        get_filename_component(SHADER_NAME ${SHADER} NAME_WE)
        set(AIR_FILE "${PROJECT_BINARY_DIR}/${SHADER_NAME}.air")


        
        # Compile each shader to an AIR file
        add_custom_command(
            OUTPUT ${AIR_FILE}
            COMMAND xcrun -sdk macosx metal -Wno-c++17-extensions ${COMMAND_INCLUDE_DIRS} -O3 -c ${SHADER} -o ${AIR_FILE}
            DEPENDS ${SHADER} ${METAL_DEPS}
            COMMENT "Compiling ${SHADER_NAME} to AIR format"
        )

        # Collect the AIR files
        list(APPEND AIR_FILES ${AIR_FILE})
    endforeach()

    set(METALLIB_FILE ${PROJECT_BINARY_DIR}/${LIB_PREFIX_NAME}.metallib)

    # Link AIR file to metallib
    add_custom_command(
        OUTPUT ${METALLIB_FILE}
        COMMAND xcrun -sdk macosx metallib ${AIR_FILES} -o ${METALLIB_FILE}
        DEPENDS ${AIR_FILES}
        COMMENT "Creating Metal library for '${LIB_PREFIX_NAME}'"
    )

    # Use xxd to convert .metallib to C++ source
    add_custom_command(
        OUTPUT ${OUTPUT_FILE}
        COMMAND xxd -i -n ${LIB_PREFIX_NAME}_metallib ${METALLIB_FILE} > ${OUTPUT_FILE}
        DEPENDS ${METALLIB_FILE}
        COMMENT "Converting Metal library for '${LIB_PREFIX_NAME}' to C++ (data) symbol"
    )
endfunction()