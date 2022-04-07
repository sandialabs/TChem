Building TChem
==============

TChem is designed and implemented using Kokkos (a performance portable parallel programming model). The code relines on **Tines** for SACADO automatic derivative (AD) types and time integration. Optionally, it uses CVODE library from SUNDIALS. The GTEST framework is used for unit tests. 

For convenience, we explain how to build the TChem code using the following environment variable that a user can modify according to their working environments.

.. code-block:: bash

   # repositories
   export TCHEM_REPOSITORY_PATH=/where/you/clone/tchem/git/repo
   export KOKKOS_REPOSITORY_PATH=/where/you/clone/kokkos/git/repo
   export TINES_REPOSITORY_PATH=/where/you/clone/tines/git/repo
   export GTEST_REPOSITORY_PATH=/where/you/clone/gtest/git/repo

   # build directories
   export TCHEM_BUILD_PATH=/where/you/build/tchem
   export KOKKOS_BUILD_PATH=/where/you/build/kokkos
   export TINES_REPOSITORY_PATH=/where/you/clone/tines/git/repo		
   export GTEST_BUILD_PATH=/where/you/build/gtest

   # install directories
   export TCHEM_INSTALL_PATH=/where/you/install/tchem
   export KOKKOS_INSTALL_PATH=/where/you/install/kokkos
   export TINES_INSTALL_PATH=/where/you/install/tines
   export GTEST_INSTALL_PATH=/where/you/install/gtest

Download Libraries
------------------

Clone Kokkos, Tines, GTEST and TChem repositories. 

.. code-block:: bash

   git clone https://github.com/kokkos/kokkos.git ${KOKKOS_REPOSITORY_PATH};
   git clone https://github.com/sandialabs/Tines.git ${TINES_REPOSITORY_PATH};   
   git clone https://github.com/google/googletest.git ${GTEST_REPOSITORY_PATH}
   git clone https://github.com/sandialabs/TChem.git ${TCHEM_REPOSITORY_PATH};

Here, we assume standard third party libraries (TPLs) e.g., BLAS and LAPACK are installed separately.

Building Libraries and TChem
----------------------------

Kokkos
~~~~~~

This example build Kokkos on Intel Sandybridge architectures and install it to ``${KOKKOS_INSTALL_PATH}``. For more details, see [Kokkos github pages](https://github.com/kokkos/kokkos).

.. code-block:: bash
		
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

To compile for NVIDIA GPUs, one can customize the following cmake script. Note that we use Kokkos nvcc_wrapper as its compiler. The architecture flag indicates that the host architecture is Intel SandyBridge and the GPU architecture is Volta 70 generation. With Kokkko 3.1, the CUDA architecture flag is optional (the script automatically detects a correct CUDA arch flag).

.. code-block:: bash
		
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

Tines
~~~~~

Tines provides batched implicit time integration solver for stiff ODE and DAE problems.

.. code-block:: bash
		
   cmake \
     -D CMAKE_INSTALL_PREFIX=${TINES_INSTALL_PATH} \
     -D CMAKE_C_COMPILER="${CC}" \
     -D CMAKE_CXX_COMPILER="${CXX}" \
     -D CMAKE_BUILD_TYPE=RELEASE \
     -D TINES_ENABLE_DEBUG=OFF \
     -D TINES_ENABLE_VERBOSE=OFF \
     -D KOKKOS_INSTALL_PATH="${KOKKOS_INSTALL_PATH}" \
     -D OPENBLAS_INSTALL_PATH="${OPENBLAS_INSTALL_PATH}" \
    ${TINES_SRC_PATH}
   make -j install

Optionally, Tines can interface to CVODE by adding ``-D SUNDIALS_INSTALL_PATH=${SUNDIALS_INSTALL_PATH}``. The code interfaces to SUNDIALS version 6.1.0 and higher. 
   
GTEST
~~~~~

We use GTEST as our testing infrastructure. With the following cmake script, the GTEST can be compiled and installed.

.. code-block:: bash
		
   cd ${GTEST_BUILD_PATH}
   cmake \
     -D CMAKE_INSTALL_PREFIX="${GTEST_INSTALL_PATH}" \
     -D CMAKE_CXX_COMPILER="${CXX}"  \
     ${GTEST_REPOSITORY_PATH}
   make -j install

TChem
~~~~~

The following example cmake script compiles TChem on host linking with the libraries described in the above e.g., kokkos, tines, and gtest. For NVIDIA GPUs, we can use the above cmake script replacing the compiler with Kokkos nvcc_wrapper by adding ``-D CMAKE_CXX_COMPILER="${KOKKOS_INSTALL_PATH}/bin/nvcc_wrapper"``.

.. code-block:: bash
		
   cd ${TCHEM_BUILD_PATH}
   cmake \
     -D CMAKE_INSTALL_PREFIX=${TCHEM_INSTALL_PATH} \
     -D CMAKE_CXX_COMPILER="${CXX}" \
     -D CMAKE_BUILD_TYPE=RELEASE \
     -D TCHEM_ENABLE_REAL_TYPE="double" \
     -D TCHEM_ENABLE_PYTHON=ON \
     -D TCHEM_ENABLE_DEBUG=OFF \
     -D TCHEM_ENABLE_VERBOSE=OFF \
     -D TCHEM_ENABLE_TEST=OFF \
     -D TCHEM_ENABLE_EXAMPLE=ON \
     -D TCHEM_ENABLE_SACADO_JACOBIAN_IGNITION_ZERO_D_REACTOR=ON \
     -D TCHEM_ENABLE_SACADO_JACOBIAN_CONSTANT_VOLUME_IGNITION_REACTOR=ON \
     -D KOKKOS_INSTALL_PATH="${KOKKOS_INSTALL_PATH}" \
     -D TINES_INSTALL_PATH="${TINES_INSTALL_PATH}" \
     -D GTEST_INSTALL_PATH="${GTEST_INSTALL_PATH}" \
   ${TCHEM_SRC_PATH}
   make -j install

TChem support both analytic and numerical Jacobians for canonical reactor models. The Jacobian types are selected in the configure time and the following options are available.

.. code-block:: bash

   # Available options are
   #   "ANALYTIC" - hand derived analytic Jacobian when available
   #   "NUMERICAL" - numerical Jacobian computed via difference method
   #   "SACADO" -  analytic Jacobian computed via SACADO AD type
   -D TCHEM_ENABLE_JACOBIAN_ATMOSPHERIC_CHEMISTRY="${OPTION}"
   -D TCHEM_ENABLE_JACOBIAN_CONSTANT_PRESSURE_IGNITION_REACTOR="${OPTION}"   
   -D TCHEM_ENABLE_JACOBIAN_CONSTANT_VOLUME_IGNITION_REACTOR="${OPTION}"
   -D TCHEM_ENABLE_JACOBIAN_PLUG_FLOW_REACTOR="${OPTION}"
   -D TCHEM_ENABLE_JACOBIAN_TRANSIENT_CONT_STIRRED_TANK_REACTOR="${OPTION}"

TODO - this does not match to current code but we need to reduce complexity. Too many options with inconsistent names.

A successful installation creates following directory structure in ``${TCHEM_INSTALL_PATH}``.

.. code-block:: bash

   - bin
     - tchem.x
     - tchem-json-test.x
   - example
     - example executables
     - data
     - runs
   - include
     - tchem
       - header files
   - lib (or lib64)	 
     - cmake
     - libtchem.a
     - pytchem.***.so

       
.. autosummary::
   :toctree: generated

   tchem
