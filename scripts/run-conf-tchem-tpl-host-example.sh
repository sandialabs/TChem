rm -f CMakeCache.txt;
rm -rf CMakeFiles

TCHEM_INSTALL_PATH=/Users/odiazib/CODE/TChem++/install
TCHEM_SRC_PATH=/Users/odiazib/CODE/TChem++/src
KOKKOS_INSTALL_PATH=/Users/odiazib/csp_gnu_bld/kokkos/install
KOKKOSKERNELS_INSTALL_PATH=/Users/odiazib/csp_gnu_bld/kokkoskernels/install

cmake \
    -D CMAKE_INSTALL_PREFIX=${TCHEM_INSTALL_PATH} \
    -D CMAKE_CXX_COMPILER=g++ \
    -D CMAKE_CXX_FLAGS="-g -lineinfo" \
    -D TCHEM_ENABLE_KOKKOS=ON \
    -D KOKKOS_INSTALL_PATH=${KOKKOS_INSTALL_PATH} \
    -D TCHEM_ENABLE_KOKKOSKERNELS=ON \
    -D KOKKOSKERNELS_INSTALL_PATH=${KOKKOSKERNELS_INSTALL_PATH} \
    -D OPENBLAS_INSTALL_PATH="/opt/local" \    
    ${TCHEM_SRC_PATH}
