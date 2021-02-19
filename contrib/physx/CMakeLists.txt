
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

if(${CMAKE_BUILD_TYPE} STREQUAL "Debug")
    set(PHYSX_BUILD_TYPE Debug)
    set(PHYSX_BUILD_OUT debug)
else()
    set(PHYSX_BUILD_TYPE Release)
    set(PHYSX_BUILD_OUT release)
endif()

ExternalProject_Add(PhysX_external
    # PhysX_4.1.0.25992954
    URL https://github.com/NVIDIAGameWorks/PhysX/archive/4050bbfdc2699dfab7edbf0393df8ff96bbe06c5.zip
    URL_HASH SHA256=f47571fdb732811336703cd4a3db2376855a9430db7ff0f7ea432ab895097da5

    PATCH_COMMAND patch -p1 < ${CMAKE_CURRENT_SOURCE_DIR}/fix_fatal_compiler_warnings.patch && cd physx && ./generate_projects.sh linux && cd ..

    SOURCE_SUBDIR physx/compiler/public

    # Extracted from the PhysX build system wrapper scripts, which supply
    # these as -D... options, but some via environment variables.
    CMAKE_ARGS
        --no-warn-unused-cli
        -DTARGET_BUILD_PLATFORM=linux
        -DPX_OUTPUT_ARCH=x86
        "-DCMAKE_PREFIX_PATH=<SOURCE_DIR>/externals/opengl-linux"
        "-DPHYSX_ROOT_DIR=<SOURCE_DIR>/physx"
        "-DCMAKE_BUILD_TYPE=${PHYSX_BUILD_TYPE}"
        "-DCMAKEMODULES_PATH=<SOURCE_DIR>/externals/cmakemodules"
        "-DPXSHARED_PATH=<SOURCE_DIR>/pxshared"
        "-DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>"
        "-DPXSHARED_INSTALL_PREFIX:PATH=<INSTALL_DIR>"
        "-DPX_OUTPUT_LIB_DIR=<SOURCE_DIR>/physx"
        "-DPX_OUTPUT_BIN_DIR=<SOURCE_DIR>/physx"
        "-DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}"

    USES_TERMINAL_BUILD ON

    BUILD_BYPRODUCTS
        "<INSTALL_DIR>/bin/linux.clang/${PHYSX_BUILD_OUT}/libPhysX_64.so"
        "<INSTALL_DIR>/bin/linux.clang/${PHYSX_BUILD_OUT}/libPhysXCommon_64.so"
        "<INSTALL_DIR>/bin/linux.clang/${PHYSX_BUILD_OUT}/libPhysXFoundation_64.so"
        "<INSTALL_DIR>/bin/linux.clang/${PHYSX_BUILD_OUT}/libPhysXPvdSDK_static_64.a"
        "<INSTALL_DIR>/bin/linux.clang/${PHYSX_BUILD_OUT}/libPhysXExtensions_static_64.a"
        "<INSTALL_DIR>/bin/linux.clang/${PHYSX_BUILD_OUT}/libPhysXCooking_64.so"
)

ExternalProject_Get_Property(PhysX_external INSTALL_DIR)

# Define an interface target for PhysX
add_library(PhysX INTERFACE)

if(${PHYSX_BUILD_TYPE} STREQUAL "Debug")
    target_compile_definitions(PhysX INTERFACE -D_DEBUG=1)
    set(OUT_DIR "${INSTALL_DIR}/bin/linux.clang/debug")
else()
    set(OUT_DIR "${INSTALL_DIR}/bin/linux.clang/release")
endif()

set(LIBRARIES
    ${OUT_DIR}/libPhysX_64.so
    ${OUT_DIR}/libPhysXPvdSDK_static_64.a
    ${OUT_DIR}/libPhysXExtensions_static_64.a
    ${OUT_DIR}/libPhysXCooking_64.so
    "-Wl,--no-as-needed"
    ${OUT_DIR}/libPhysXCommon_64.so
    "-Wl,--as-needed"
    ${OUT_DIR}/libPhysXFoundation_64.so
)

target_link_libraries(PhysX INTERFACE ${LIBRARIES})

add_dependencies(PhysX PhysX_external)

target_include_directories(PhysX INTERFACE ${INSTALL_DIR}/include)

install(FILES
    ${OUT_DIR}/libPhysX_64.so
    ${OUT_DIR}/libPhysXFoundation_64.so
    ${OUT_DIR}/libPhysXPvdSDK_static_64.a
    ${OUT_DIR}/libPhysXExtensions_static_64.a
    ${OUT_DIR}/libPhysXCooking_64.so
    ${OUT_DIR}/libPhysXCommon_64.so
    DESTINATION lib
)

install(DIRECTORY
    ${INSTALL_DIR}/include/
    DESTINATION include
)
