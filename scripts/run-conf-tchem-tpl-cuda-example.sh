rm -f CMakeCache.txt;
rm -rf CMakeFiles

TCHEM_INSTALL_PATH=/your/tchem/install/path
TCHEM_SRC_PATH=/your/tchem/src/path
KOKKOS_INSTALL_PATH=/your/kokkos/install/path
KOKKOS_NVCC_WRAPPER=${KOKKOS_INSTALL_PATH}/bin/nvcc_wrapper
MATHUTILS_INSTALL_PATH=/your/mathutils/install/path
GTEST_INSTALL_PATH=/your/gtest/install/path

cmake \
    -D CMAKE_INSTALL_PREFIX=${TCHEM_INSTALL_PATH} \
    -D CMAKE_CXX_COMPILER=${KOKKOS_NVCC_WRAPPER} \
    -D CMAKE_CXX_FLAGS="-g -lineinfo" \
    -D TCHEM_ENABLE_VERBOSE=OFF \
    -D TCHEM_ENABLE_TEST=ON \
    -D TCHEM_ENABLE_EXAMPLE=ON \
    -D KOKKOS_INSTALL_PATH=${KOKKOS_INSTALL_PATH} \
    -D MATHUTILS_INSTALL_PATH=${MATHUTILS_INSTALL_PATH} \
    -D GTEST_INSTALL_PATH=${GTEST_INSTALL_PATH} \
    ${TCHEM_SRC_PATH}
