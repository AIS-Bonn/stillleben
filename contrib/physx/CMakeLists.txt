
# Download and compile PhysX

include(ExternalProject)


# sample command line by physx build script:
# cmake "/home/max/projects/phd/stillleben/cpp_build/contrib/physx/PhysX-prefix/src/PhysX/physx/compiler/public"
# -G "Unix Makefiles" -DTARGET_BUILD_PLATFORM=linux -DPX_OUTPUT_ARCH=x86 -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++
# --no-warn-unused-cli -DCMAKE_PREFIX_PATH="/home/max/projects/phd/stillleben/cpp_build/contrib/physx/PhysX-prefix/src/PhysX/physx/../externals/opengl-linux"
# -DPHYSX_ROOT_DIR="/home/max/projects/phd/stillleben/cpp_build/contrib/physx/PhysX-prefix/src/PhysX/physx"
# -DPX_OUTPUT_LIB_DIR="/home/max/projects/phd/stillleben/cpp_build/contrib/physx/PhysX-prefix/src/PhysX/physx"
# -DPX_OUTPUT_BIN_DIR="/home/max/projects/phd/stillleben/cpp_build/contrib/physx/PhysX-prefix/src/PhysX/physx"
# -DPX_BUILDSNIPPETS=TRUE -DPX_BUILDSAMPLES=TRUE -DPX_GENERATE_STATIC_LIBRARIES=TRUE
# -DCMAKE_INSTALL_PREFIX="/home/max/projects/phd/stillleben/cpp_build/contrib/physx/PhysX-prefix/src/PhysX/physx/install/linux/PhysX"
# -DCMAKE_BUILD_TYPE=release

ExternalProject_Add(PhysX_external
    # PhysX_4.1.0.25992954
    URL https://github.com/NVIDIAGameWorks/PhysX/archive/4050bbfdc2699dfab7edbf0393df8ff96bbe06c5.zip
    URL_HASH SHA256=f47571fdb732811336703cd4a3db2376855a9430db7ff0f7ea432ab895097da5

    PATCH_COMMAND cd physx && ./generate_projects.sh linux && cd ..

    SOURCE_SUBDIR physx/compiler/public

    # Extracted from the PhysX build system wrapper scripts, which supply
    # these as -D... options, but some via environment variables.
    CMAKE_ARGS
        --no-warn-unused-cli
        -DTARGET_BUILD_PLATFORM=linux
        -DPX_OUTPUT_ARCH=x86
        "-DCMAKE_PREFIX_PATH=<SOURCE_DIR>/externals/opengl-linux"
        "-DPHYSX_ROOT_DIR=<SOURCE_DIR>/physx"
        "-DCMAKE_BUILD_TYPE=Release"
        "-DCMAKEMODULES_PATH=<SOURCE_DIR>/externals/cmakemodules"
        "-DPXSHARED_PATH=<SOURCE_DIR>/pxshared"
        "-DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>"
        "-DPXSHARED_INSTALL_PREFIX:PATH=<INSTALL_DIR>"
        "-DPX_OUTPUT_LIB_DIR=<SOURCE_DIR>/physx"
        "-DPX_OUTPUT_BIN_DIR=<SOURCE_DIR>/physx"

    USES_TERMINAL_BUILD ON

    BUILD_BYPRODUCTS
        "<INSTALL_DIR>/bin/linux.clang/release/libPhysX_64.so"
        "<INSTALL_DIR>/bin/linux.clang/release/libPhysXCommon_64.so"
        "<INSTALL_DIR>/bin/linux.clang/release/libPhysXFoundation_64.so"
        "<INSTALL_DIR>/bin/linux.clang/release/libPhysXPvdSDK_static_64.a"
        "<INSTALL_DIR>/bin/linux.clang/release/libPhysXExtensions_static_64.a"
        "<INSTALL_DIR>/bin/linux.clang/release/libPhysXCooking_64.so"
)

ExternalProject_Get_Property(PhysX_external INSTALL_DIR)

add_library(PhysX INTERFACE)
set(LIBRARIES
    ${INSTALL_DIR}/bin/linux.clang/release/libPhysX_64.so
    ${INSTALL_DIR}/bin/linux.clang/release/libPhysXFoundation_64.so
    ${INSTALL_DIR}/bin/linux.clang/release/libPhysXPvdSDK_static_64.a
    ${INSTALL_DIR}/bin/linux.clang/release/libPhysXExtensions_static_64.a
    ${INSTALL_DIR}/bin/linux.clang/release/libPhysXCooking_64.so
    "-Wl,--no-as-needed"
    ${INSTALL_DIR}/bin/linux.clang/release/libPhysXCommon_64.so
    "-Wl,--as-needed"
)

target_link_libraries(PhysX INTERFACE ${LIBRARIES})

add_dependencies(PhysX PhysX_external)

target_include_directories(PhysX INTERFACE ${INSTALL_DIR}/include)

install(FILES
    ${INSTALL_DIR}/bin/linux.clang/release/libPhysX_64.so
    ${INSTALL_DIR}/bin/linux.clang/release/libPhysXFoundation_64.so
    ${INSTALL_DIR}/bin/linux.clang/release/libPhysXPvdSDK_static_64.a
    ${INSTALL_DIR}/bin/linux.clang/release/libPhysXExtensions_static_64.a
    ${INSTALL_DIR}/bin/linux.clang/release/libPhysXCooking_64.so
    ${INSTALL_DIR}/bin/linux.clang/release/libPhysXCommon_64.so
    DESTINATION lib
)
