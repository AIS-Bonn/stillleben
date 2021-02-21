
add_executable(tests
    basic.cpp
    image_loader.cpp
    image_saver.cpp
    runner.cpp
)

target_compile_definitions(tests
    PRIVATE -DPATH_TO_SOURCES="${CMAKE_CURRENT_SOURCE_DIR}/.."
)

target_link_libraries(tests
    stillleben
)
