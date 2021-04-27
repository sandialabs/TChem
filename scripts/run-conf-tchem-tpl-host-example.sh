rm -f CMakeCache.txt;
rm -rf CMakeFiles

TCHEM_INSTALL_PATH=/Users/odiazib/CODE/TChem++/install
TCHEM_SRC_PATH=/Users/odiazib/CODE/TChem++/src
KOKKOS_INSTALL_PATH=/Users/odiazib/csp_gnu_bld/kokkos/install
MATHUTILS_INSTALL_PATH=/Users/odiazib/csp_gnu_bld/mathutils/install
GTEST_INSTALL_PATH=/Users/odiazib/csp_gnu_bld/gtest/install

cmake \
    -D CMAKE_INSTALL_PREFIX=${TCHEM_INSTALL_PATH} \
    -D CMAKE_CXX_COMPILER=g++ \
    -D CMAKE_CXX_FLAGS="-g -lineinfo" \
    -D TCHEM_ENABLE_VERBOSE=OFF \
    -D TCHEM_ENABLE_TEST=ON \
    -D TCHEM_ENABLE_EXAMPLE=ON \
    -D KOKKOS_INSTALL_PATH=${KOKKOS_INSTALL_PATH} \
    -D MATHUTILS_INSTALL_PATH=${MATHUTILS_INSTALL_PATH} \
    -D GTEST_INSTALL_PATH=${GTEST_INSTALL_PATH} \
    ${TCHEM_SRC_PATH}
