
find_package(Python3 COMPONENTS Interpreter Development)

if(NOT Python3_FOUND)
    message(WARNING "Could not find Python 3, not building python bindings")
    return()
endif()

# execute our helper that gets us the build flags
execute_process(
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/utils/find_torch.py
    RESULT_VARIABLE FIND_TORCH_RESULT
    OUTPUT_VARIABLE FIND_TORCH_OUTPUT
)

if(NOT ${FIND_TORCH_RESULT} EQUAL 0)
    message(WARNING "Could not find PyTorch, not building python bindings")
    return()
endif()

# Can't execute cmake code directly, so write to file and include()
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/pytorch_paths.cmake
    "${FIND_TORCH_OUTPUT}"
)
include(${CMAKE_CURRENT_BINARY_DIR}/pytorch_paths.cmake)

message(STATUS "Found PyTorch ${TORCH_VERSION}.")
message(STATUS " - CUDA support: ${TORCH_USE_CUDA}")

add_library(torch INTERFACE)
target_include_directories(torch INTERFACE
    ${TORCH_INCLUDE_DIRS} ${Python3_INCLUDE_DIRS}
)

if(${TORCH_VERSION} VERSION_GREATER_EQUAL 1.4.0)
    # Find core libraries & link them directly
    find_library(C10_LIBRARY c10 PATHS ${TORCH_LIBRARY_DIRS})
    find_library(TORCH_LIBRARY torch PATHS ${TORCH_LIBRARY_DIRS})
    find_library(TORCH_PYTHON_LIBRARY torch_python PATHS ${TORCH_LIBRARY_DIRS})
    message(STATUS " - c10 lib: ${C10_LIBRARY}")
    message(STATUS " - torch lib: ${TORCH_LIBRARY}")
    message(STATUS " - torch_python lib: ${TORCH_PYTHON_LIBRARY}")

    target_link_libraries(torch INTERFACE
        ${C10_LIBRARY}
        ${TORCH_LIBRARY}
        ${TORCH_PYTHON_LIBRARY}
    )
endif()

add_library(stillleben_python MODULE
    src/bridge.cpp
    src/py_context.cpp
    src/py_magnum.cpp
    src/py_mesh.cpp
    src/py_object.cpp
)
target_link_libraries(stillleben_python
    PUBLIC
        stillleben
        torch
)

if(TORCH_USE_CUDA)
    target_compile_definitions(stillleben_python PRIVATE HAVE_CUDA=1)
endif()

install(TARGETS stillleben_python
    LIBRARY DESTINATION lib
)

# diff packages
if(TORCH_USE_CUDA)
    find_package(CUDA QUIET)

    find_program(CMAKE_CUDA_COMPILER nvcc HINTS ${CUDA_TOOLKIT_ROOT_DIR}/bin)

    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
    else()
        message(WARNING "PyTorch was built with CUDA support, but I couldn't find a CUDA compiler...")
    endif()
endif()

if(TORCH_USE_CUDA AND CMAKE_CUDA_COMPILER)
    message(STATUS "Building Differentiation Modules")

    find_package(CUDA REQUIRED)
    add_library(stillleben_diff_python MODULE
        src/bridge_diff.cpp
        src/diff.cu
        src/diff.h
    )

    target_include_directories(stillleben_diff_python
        PRIVATE ${TORCH_INCLUDE_DIRS} ${Python3_INCLUDE_DIRS}
    )

    target_link_libraries(stillleben_diff_python
        PUBLIC
            Corrade::Utility
            Corrade::Containers
            torch
    )
    target_compile_definitions(stillleben_diff_python PRIVATE HAVE_CUDA=1)

    # nvcc gets confused by our compiler flags
    get_target_property(_target_cxx_flags stillleben_diff_python COMPILE_OPTIONS)
    list(REMOVE_ITEM _target_cxx_flags "-Wl,--no-as-needed")
    if(${CMAKE_GENERATOR} STREQUAL "Ninja")
        if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
            list(REMOVE_ITEM _target_cxx_flags "-fdiagnostics-color=always")
        elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
            list(REMOVE_ITEM _target_cxx_flags "-fcolor-diagnostics")
        endif()
    endif()

    set_target_properties(stillleben_diff_python PROPERTIES
        COMPILE_OPTIONS "${_target_cxx_flags}"
    )

    install(TARGETS stillleben_diff_python
        LIBRARY DESTINATION lib
    )
else()
    message(STATUS "CUDA not available, skipping differentiation modules")
endif()
