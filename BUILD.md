# Building TChem

TChem is designed and implemented using [Kokkos](https://github.com/kokkos/kokkos.git) (a performance portable parallel programming model) and requires a set of third-party libraries including [BLAS/LAPACK](http://www.openblas.net), [YAML](https://github.com/jbeder/yaml-cpp) and [Tines](https://github.com/sandialabs/Tines.git).
As most third-party libraries (TPLs) can be installed by using a package manager i.e., macports on OSX and yum/apt on Linux, we mainly explain how to build Kokkos, Tines, and TChem.
To simplify the TChem build process we define the following environment variables. These should be tailored to specific working environments by the user.


```
/// repositories
export TCHEM_REPOSITORY_PATH=/where/you/clone/tchem/git/repo
export KOKKOS_REPOSITORY_PATH=/where/you/clone/kokkos/git/repo
export TINES_REPOSITORY_PATH=/where/you/clone/tines/git/repo

/// build directories
export TCHEM_BUILD_PATH=/where/you/build/tchem
export KOKKOS_BUILD_PATH=/where/you/build/kokkos
export TINES_BUILD_PATH=/where/you/build/tines

/// install directories
export TCHEM_INSTALL_PATH=/where/you/install/tchem
export KOKKOS_INSTALL_PATH=/where/you/install/kokkos
export TINES_INSTALL_PATH=/where/you/install/tines

/// third party libraries install path
export GTEST_INSTALL_PATH=/where/you/install/gtest
export OPENBLAS_INSTALL_PATH=/where/you/install/openblas
export LAPACKE_INSTALL_PATH=/where/you/install/lapacke
export YAML_INSTALL_PATH=/where/you/intall/yaml-cpp
```

## Download Libraries


Please see below the syntax for downloading the Kokkos, Tines, and TChem repositories.

```
git clone https://github.com/sandialabs/Tines.git  ${TINES_REPOSITORY_PATH};
git clone https://github.com/kokkos/kokkos.git ${KOKKOS_REPOSITORY_PATH};
git clone https://github.com/sandialabs/TChem.git ${TCHEM_REPOSITORY_PATH};
```
In the next section we present the workflow for building TChem and the pre-requisite TPLs.

## Building Libraries and Configuring TChem

### Kokkos

Kokkos requires a ``Kokkos_ARCH_XXXX`` flag to perform optimization based on the given architecture information. This particular example script is based on Intel Sandy-Bridge i.e., ``-D Kokkos_ENABLE_SNB=ON`` and will install the library to ``${KOKKOS_INSTALL_PATH}``. For more details, see [Kokkos github pages](https://github.com/kokkos/kokkos).

```
cd ${KOKKOS_BUILD_PATH}
cmake \
    -D CMAKE_INSTALL_PREFIX="${KOKKOS_INSTALL_PATH}" \
    -D CMAKE_CXX_COMPILER="${CXX}"  \
    -D Kokkos_ENABLE_SERIAL=ON \
    -D Kokkos_ENABLE_OPENMP=ON \
    -D Kokkos_ENABLE_DEPRECATED_CODE=OFF \
    -D Kokkos_ARCH_SNB=ON \
    ${KOKKOS_REPOSITORY_PATH}
make -j install
export KOKKOS_CXX_COMPILER=${CXX}
```

To compile for NVIDIA GPUs, one can customize the following cmake script. Note that we use Kokkos ``nvcc_wrapper`` as its compiler. The architecture flag indicates that the host architecture is Intel Sandy-Bridge and the GPU architecture is Volta 70 generation. With Kokkos 3.1, the CUDA architecture flag is optional (the script automatically detects the correct CUDA arch flag).
```
cd ${KOKKOS_BUILD_PATH}
cmake \
    -D CMAKE_INSTALL_PREFIX="${KOKKOS_INSTALL_PATH}" \
    -D CMAKE_CXX_COMPILER="${KOKKOS_REPOSITORY_PATH}/bin/nvcc_wrapper"  \
    -D Kokkos_ENABLE_SERIAL=ON \
    -D Kokkos_ENABLE_OPENMP=ON \
    -D Kokkos_ENABLE_CUDA:BOOL=ON \
    -D Kokkos_ENABLE_CUDA_UVM:BOOL=OFF \
    -D Kokkos_ENABLE_CUDA_LAMBDA:BOOL=ON \
    -D Kokkos_ENABLE_DEPRECATED_CODE=OFF \
    -D Kokkos_ARCH_VOLTA70=ON \
    -D Kokkos_ARCH_SNB=ON \
    ${KOKKOS_REPOSITORY_PATH}
make -j install
export KOKKOS_CXX_COMPILER=${KOKKOS_INSTALL_PATH}/bin/nvcc_wrapper
```
For GPUs, we note that the compiler is switched to ``nvcc_wrapper`` by adding ``-D CMAKE_CXX_COMPILER="${KOKKOS_INSTALL_PATH}/bin/nvcc_wrapper"``. To use the same compiler setup for other libraries, we export an environmental variable ``KOKKOS_CXX_COMPILER`` according to its target architecture.  


### Tines
Compiling Tines follows the Kokkos configuration information available under ``${KOKKOS_INSTALL_PATH}``. The *OpenBLAS* and *LAPACKE* libraries are required on a host device providing an optimized version of dense linear algebra library. With an Intel compiler, one can replace these libraries with Intel MKL by adding the flag ``TINES_ENABLE_MKL=ON`` instead of using *OpenBLAS* and *LAPACKE*. On Mac OSX, we use the *OpenBLAS* library managed by **MacPorts**. This version has different header names and we need to distinguish this version of the code from others which are typically used on Linux distributions. To discern the two version of the code, cmake looks for "cblas\_openblas.h" to tell that the installed version is from MacPorts. This mechanism can be broken if MacPorts' *OpenBLAS* is changed later. Furthermore, the MacPorts *OpenBLAS* version include LAPACKE interface and one can remove  ``LAPACKE_INSTALL_PATH`` from the configure script. Additionally, the yaml-cpp (version 0.6.3) is used for parsing input files in YAML format.


```
cmake \
    -D CMAKE_INSTALL_PREFIX=${TINES_INSTALL_PATH} \
    -D CMAKE_CXX_COMPILER="${KOKKOS_CXX_COMPILER}" \
    -D CMAKE_CXX_FLAGS="-g" \
    -D TINES_ENABLE_DEBUG=OFF \
    -D TINES_ENABLE_VERBOSE=OFF \
    -D TINES_ENABLE_TEST=ON \
    -D TINES_ENABLE_EXAMPLE=ON \
    -D KOKKOS_INSTALL_PATH="${KOKKOS_INSTALL_PATH} \
    -D GTEST_INSTALL_PATH="${GTEST_INSTALL_PATH" \
    -D OPENBLAS_INSTALL_PATH="${OPENBLAS_INSTALL_PATH}" \
    -D LAPACKE_INSTALL_PATH="${LAPACKE_INSTALL_PATH}" \
    -D YAML_INSTALL_PATH="${YAML_INSTALL_PATH}" \
    ${TINES_REPOSITORY_PATH}/src
make -j install
```

### TChem

The following example cmake script compiles TChem on host linking with the libraries described in the above e.g., KOKKOS, Tines, GTEST, and OpenBLAS.
```
cd ${TCHEM_BUILD_PATH}
cmake \
    -D CMAKE_INSTALL_PREFIX="${TCHEM_INSTALL_PATH}" \
    -D CMAKE_CXX_COMPILER="${KOKKOS_CXX_COMPILER}" \
    -D CMAKE_BUILD_TYPE=RELEASE \
    -D TCHEM_ENABLE_VERBOSE=OFF \
    -D TCHEM_ENABLE_TEST=ON \
    -D TCHEM_ENABLE_EXAMPLE=ON \
    -D KOKKOS_INSTALL_PATH="${KOKKOS_INSTALL_PATH}" \
    -D TINES_INSTALL_PATH="${TINES_INSTALL_PATH}" \
    -D GTEST_INSTALL_PATH="${GTEST_INSTALL_PATH}" \
    -D TCHEM_ENABLE_PYTHON=OFF \
    ${TCHEM_REPOSITORY_PATH}/src
make -j install
```
Optionally, a user can enable the Python interface with ``-D TCHEM_ENABLE_PYTHON=ON``. The Python interface of TChem is ported via [pybind11](https://pybind11.readthedocs.io/en/stable) and the code is distributed together with TChem.

### TChem Main executable

TChem main executable uses/produces inputs/outputs in json format. This code is under construction, but it will replace all executables in the example directory. It uses boost 1.75 to parse json files, so boost is required. TChem's cmake implementation will search for boost in the system directory. If boost is not installed in this directory, export its installation path with ``export BOOST_ROOT=/where/boost/is/installed``. On the other hand, if one wants to continue using the executables from the example directory and do not want to build/install the main executable then use the cmake flag ``-D TCHEM_ENABLE_MAIN=OFF``; in this case boost is not required.

### Script to Build and Install TChem and its TPLs

We put together a script, ``${TCHEM_REPOSITORY_PATH}/scripts/master_bld.sh``, that clones, builds, and installs TChem, and the required TPLs (Kokkos, Tines, OpenBLAS, GTEST, and yaml-cpp). To use this script, update the compiler information for the C compiler,``MY_CC``, the C++ compiler ``MY_CXX``, and Fortran compiler ``MY_FC``. One can also choose whether to build/install the entire framework for GPU computations, i.e. ``CUDA``. These variables are located at the top of the script.    
