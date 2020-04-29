
add_library(VHACD SHARED
    src/aabbtree.cpp
    src/btAlignedAllocator.cpp
    src/btConvexHullComputer.cpp
    src/FloatMath.cpp
    src/SimpleJobSystem.cpp
    src/VHACD-ASYNC.cpp
    src/VHACD.cpp
    src/vhacdICHull.cpp
    src/vhacdManifoldMesh.cpp
    src/vhacdMesh.cpp
    src/vhacdRaycastMesh.cpp
    src/vhacdVolume.cpp
)
target_include_directories(VHACD PUBLIC
    include
)

install(TARGETS VHACD
    LIBRARY DESTINATION lib
)
