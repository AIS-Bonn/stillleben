
find_package(Python COMPONENTS Interpreter Development)

if(NOT Python_FOUND)
    message(WARNING "Could not find Python, not building python bindings")
    return()
endif()

# execute our helper that gets us the build flags
execute_process(
    COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/utils/find_torch.py
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

Python_add_library(stillleben_python MODULE
    src/bridge.cpp
)
target_include_directories(stillleben_python
    PRIVATE ${TORCH_INCLUDE_DIRS}
)
target_link_libraries(stillleben_python
    PUBLIC
        Python::Python
        stillleben
)
