# mac osx setup
export TCHEM_DEMO_ROOT=${PWD}
export JFLAG=4
export CXX=clang++-mp-9.0
export CC=clang-mp-9.0

#
# set paths
#
export TCHEM_ROOT_PATH=${TCHEM_DEMO_ROOT}/tchem
export TCHEM_INSTALL_PATH=${TCHEM_DEMO_ROOT}/demo

export GTEST_ROOT_PATH=${TCHEM_DEMO_ROOT}/gtest
export OPENBLAS_ROOT_PATH=${TCHEM_DEMO_ROOT}/openblas
export KOKKOS_ROOT_PATH=${TCHEM_DEMO_ROOT}/kokkos
export MATHUTILS_ROOT_PATH=${TCHEM_DEMO_ROOT}/mathutils

# this is for Macport
#export LIBRARY_PATH=${LIBRARY_PTH}:/opt/local/lib

#
# clone repository
#
git clone https://github.com/google/googletest.git ${GTEST_ROOT_PATH}/master
git clone https://github.com/xianyi/OpenBLAS.git ${OPENBLAS_ROOT_PATH}/master
git clone https://github.com/kokkos/kokkos.git ${KOKKOS_ROOT_PATH}/master
git clone kyukim@getz.ca.sandia.gov:/home/gitroot/math_utils ${MATHUTILS_ROOT_PATH}/master
git clone kyukim@getz.ca.sandia.gov:/home/gitroot/TChem++ ${TCHEM_ROOT_PATH}/master
cd ${TCHEM_ROOT_PATH}/master;
#git checkout v2.0.0;
git checkout --track origin kyungjoo-develop
git pull
cd ${TCHEM_DEMO_ROOT}

# compile gtest
cd ${GTEST_ROOT_PATH}; mkdir build; cd build;

cmake \
    -D CMAKE_CXX_COMPILER=${CXX} \
    -D CMAKE_INSTALL_PREFIX=${TCHEM_DEMO_ROOT}/install \
    ${GTEST_ROOT_PATH}/master
make -j${JFLAG} install
rm -rf ${GTEST_ROOT_PATH}/build

# compile openblas
cd ${OPENBLAS_ROOT_PATH}; mkdir build; cd build;

cmake \
    -D CMAKE_INSTALL_PREFIX=${TCHEM_DEMO_ROOT}/install \
    -D CMAKE_C_COMPILER=${CC} \
    -D CMAKE_EXE_LINKER_FLAGS="" \
    -D CMAKE_BUILD_TYPE=RELEASE \
    -D NOFORTRAN=ON \
    -D USE_THREAD=OFF \
    ${OPENBLAS_ROOT_PATH}/master

make -j${JFLAG} install
rm -rf ${OPENBLAS_ROOT_PATH}/build

# compile kokkos
cd ${KOKKOS_ROOT_PATH}; mkdir build; cd build;

cmake \
    -D CMAKE_INSTALL_PREFIX=${TCHEM_DEMO_ROOT}/install \
    -D CMAKE_CXX_COMPILER=${CXX}  \
    -D CMAKE_CXX_FLAGS="-g" \
    -D Kokkos_ENABLE_SERIAL=ON \
    -D Kokkos_ENABLE_OPENMP=ON \
    -D Kokkos_ENABLE_DEPRECATED_CODE=OFF \
    -D Kokkos_ARCH_HSW=ON \
    -D Kokkos_ENABLE_TESTS=OFF \
    ${KOKKOS_ROOT_PATH}/master
make -j${JFLAG} install
rm -rf ${KOKKOS_ROOT_PATH}/build

# compile kokkoskernels

cd ${MATHUTILS_ROOT_PATH}; mkdir build; cd build;

cmake \
    -D CMAKE_INSTALL_PREFIX=${TCHEM_DEMO_ROOT}/install \
    -D CMAKE_CXX_COMPILER=${CXX} \
    -D CMAKE_CXX_FLAGS="-g" \
    -D CMAKE_EXE_LINKER_FLAGS="" \
    -D CMAKE_BUILD_TYPE=RELEASE \
    -D MATHUTILS_ENABLE_DEBUG=OFF \
    -D MATHUTILS_ENABLE_VERBOSE=ON \
    -D MATHUTILS_ENABLE_KOKKOS=ON \
    -D MATHUTILS_ENABLE_TEST=ON \
    -D MATHUTILS_ENABLE_EXAMPLE=ON \
    -D KOKKOS_INSTALL_PATH=${TCHEM_DEMO_ROOT}/install \
    -D GTEST_INSTALL_PATH=${TCHEM_DEMO_ROOT}/install \
    -D OPENBLAS_INSTALL_PATH=${TCHEM_DEMO_ROOT}/install \
    ${MATHUTILS_ROOT_PATH}/master
make -j${JFLAG} install
rm -rf ${MATHUTILS_ROOT_PATH}/build

# compile tchem
cd ${TCHEM_ROOT_PATH}; mkdir build; cd build;

cmake \
    -D CMAKE_INSTALL_PREFIX=${TCHEM_INSTALL_PATH} \
    -D CMAKE_CXX_COMPILER=${CXX} \
    -D CMAKE_CXX_FLAGS="-g" \
    -D CMAKE_EXE_LINKER_FLAGS="" \
    -D CMAKE_BUILD_TYPE=RELEASE \
    -D TCHEM_ENABLE_VERBOSE=OFF \
    -D TCHEM_ENABLE_TEST=ON \
    -D TCHEM_ENABLE_EXAMPLE=ON \
    -D TCHEM_ENABLE_PROBLEMS_NUMERICAL_JACOBIAN=OFF \
    -D KOKKOS_INSTALL_PATH=${TCHEM_DEMO_ROOT}/install \
    -D GTEST_INSTALL_PATH=${TCHEM_DEMO_ROOT}/install \
    -D MATHUTILS_INSTALL_PATH=${TCHEM_DEMO_ROOT}/install \
    ${TCHEM_ROOT_PATH}/master/src

make -j${jflag} install

# back to root directory
cd ${TCHEM_DEMO_ROOT}

echo 'TChem demo compiling is done'
