/* =====================================================================================
TChem version 2.0
Copyright (2020) NTESS
https://github.com/sandialabs/TChem

Copyright 2020 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains
certain rights in this software.

This file is part of TChem. TChem is open source software: you can redistribute it
and/or modify it under the terms of BSD 2-Clause License
(https://opensource.org/licenses/BSD-2-Clause). A copy of the licese is also
provided under the main directory

Questions? Contact Cosmin Safta at <csafta@sandia.gov>, or
           Kyungjoo Kim at <kyukim@sandia.gov>, or
           Oscar Diaz-Ibarra at <odiazib@sandia.gov>

Sandia National Laboratories, Livermore, CA, USA
===================================================================================== */
#ifndef __TCHEM_UTIL_HPP__
#define __TCHEM_UTIL_HPP__

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <random>
#include <set>
#include <string>
#include <tuple>
#include <vector>

#include <cassert>
#include <cmath>
#include <ctime>
#include <limits>

#include <complex>

/// kokkos
#include "Kokkos_Complex.hpp"
#include "Kokkos_Core.hpp"
#include "Kokkos_DualView.hpp"
#include "impl/Kokkos_Timer.hpp"

/// kokkoskernels
#include "Kokkos_ArithTraits.hpp"

/// tchem configure header
#include "TChem_Config.hpp"

/// blas
#if defined(TCHEM_ENABLE_TPL_MKL)
#include "mkl.h"
#else
#if defined(TCHEM_ENABLE_TPL_OPENBLAS)
#if defined(TCHEM_ENABLE_TPL_OPENBLAS_CBLAS_HEADER)
#include "cblas_openblas.h"
#else
#include "cblas.h"
#endif
#endif
#endif

/// lapacke
#if defined(TCHEM_ENABLE_TPL_MKL)
#include "mkl.h"
#else
#if defined(TCHEM_ENABLE_TPL_LAPACKE)
#include "lapacke.h"
#endif
#endif

namespace TChem {

/// exec spaces
using exec_space = Kokkos::DefaultExecutionSpace;
using host_exec_space = Kokkos::DefaultHostExecutionSpace;

template<typename ExecSpace>
struct UseThisTeamPolicy
{
  using type = Kokkos::TeamPolicy<ExecSpace, Kokkos::Schedule<Kokkos::Dynamic> >;
};

/// type defs
using real_type = double;
using ordinal_type = int;
using size_type = size_t;

/// temporary testing/debugging only
//#define TCHEM_ENABLE_SERIAL_TEST_OUTPUT 1

/// control parameters
#if defined(TCHEM_ENABLE_VERBOSE)
constexpr bool verboseEnabled = true;
#else
constexpr bool verboseEnabled = false;
#endif

#if defined(TCHEM_ENABLE_DEBUG)
#if defined(KOKKOS_ACTIVE_EXECUTION_MEMORY_SPACE_HOST)
#define TCHEM_CHECK_ERROR(err, msg)                                            \
  if (err)                                                                     \
    std::logic_error(msg);
#else
#define TCHEM_CHECK_ERROR(err, msg)                                            \
  if (err)                                                                     \
    printf("%s\n", msg);
#endif
#else
#define TCHEM_CHECK_ERROR(err, msg)
#endif

/// parameters

/**
 * \def TCSMALL
 * Threshold for various numerical estimates
 */
constexpr real_type TCSMALL = 1.e-12;

// 2*pi
constexpr real_type DPI = 6.283185307179586;

constexpr real_type PI = 3.141592653589793;
/**
 * \def LENGTHOFELEMNAME
 * Maximum number of characters for element names
 */
constexpr ordinal_type LENGTHOFELEMNAME = 3;

/**
 * \def LENGTHOFSPECNAME
 *  Maximum number of characters for species names
 */
constexpr ordinal_type LENGTHOFSPECNAME = 18;

/**
 * \def NUMBEROFELEMINSPEC
 * Maximum number of (different) elements that compose a species
 */
constexpr ordinal_type NUMBEROFELEMINSPEC = 5;

/**
 * \def NTHRDBMAX
 * Maximum number of third body efficiencies
 */
constexpr ordinal_type NTHRDBMAX = 10;

/**
 * \def NPLOGMAX
 * Maximum number of interpolation ranges for PLOG
 */
constexpr ordinal_type NPLOGMAX = 10;

/**
 * \def NTH9RNGMAX
 * Maximum number of temperature ranges for 9-coefficients NASA polynomials
 */
constexpr ordinal_type NTH9RNGMAX = 5;

/**
 * \def NSPECREACMAX
 * Maximum number of reactant or product species in a reaction
 */
constexpr ordinal_type NSPECREACMAX = 6;

/**
 * \def REACBALANCE
 * Treshold for checking reaction balance with
 *        real stoichiometric coefficients
 */
constexpr real_type REACBALANCE = 1.e-4;

/**
 * \def RUNIV
 * Universal gas constant \f$J/(mol\cdot K)\f$
 */
// constexpr real_type RUNIV = 8.3144621;
constexpr real_type RUNIV = 8.314472;

/**
 * \def NAVOG
 * Avogadro's number
 */
constexpr real_type NAVOG = 6.02214179E23;

/**
 * \def ATMPA
 * Standard atmospheric pressure \f$[Pa]\f$
 */
constexpr real_type ATMPA = 101325.0;

/**
 * \def CALJO
 * Conversion from calories to Joule
 */
constexpr real_type CALJO = 4.184;

/**
 * \def KBOLT
 * Boltzmann's constant \f$(k_B)\f$ \f$[JK^{-1}]\f$
 */
constexpr real_type KBOLT = 1.3806504E-23;

/**
 * \def EVOLT
 * electron volt (eV) unit \f$[J]\f$
 */
constexpr real_type EVOLT = 1.60217653E-19;

/// time marching data structure
struct TimeAdvance
{
  real_type _tbeg, _tend;
  real_type _dt, _dtmin, _dtmax;
  ordinal_type _max_num_newton_iterations;
  ordinal_type _num_time_iterations_per_interval;
};

/// time tolerence; real_type_2d_view numberOfTimeODEs x 2 (atol,rtol)
/// newton tolerence; real_type_1d_view (atol, rtol)
using time_advance_type = TimeAdvance;
using time_advance_type_0d_dual_view =
  Kokkos::DualView<time_advance_type, Kokkos::LayoutRight, exec_space>;
using time_advance_type_1d_dual_view =
  Kokkos::DualView<time_advance_type*, Kokkos::LayoutRight, exec_space>;

using time_advance_type_0d_view =
  typename time_advance_type_0d_dual_view::t_dev;
using time_advance_type_1d_view =
  typename time_advance_type_1d_dual_view::t_dev;

using time_advance_type_0d_view_host =
  typename time_advance_type_0d_dual_view::t_host;
using time_advance_type_1d_view_host =
  typename time_advance_type_1d_dual_view::t_host;

/// view
using real_type_0d_dual_view =
  Kokkos::DualView<real_type, Kokkos::LayoutRight, exec_space>;
using real_type_1d_dual_view =
  Kokkos::DualView<real_type*, Kokkos::LayoutRight, exec_space>;
using real_type_2d_dual_view =
  Kokkos::DualView<real_type**, Kokkos::LayoutRight, exec_space>;
using real_type_3d_dual_view =
  Kokkos::DualView<real_type***, Kokkos::LayoutRight, exec_space>;

using ordinal_type_0d_dual_view =
  Kokkos::DualView<ordinal_type, Kokkos::LayoutRight, exec_space>;
using ordinal_type_1d_dual_view =
  Kokkos::DualView<ordinal_type*, Kokkos::LayoutRight, exec_space>;
using ordinal_type_2d_dual_view =
  Kokkos::DualView<ordinal_type**, Kokkos::LayoutRight, exec_space>;
using ordinal_type_3d_dual_view =
  Kokkos::DualView<ordinal_type***, Kokkos::LayoutRight, exec_space>;

template<int S>
using string_type_1d_dual_view =
  Kokkos::DualView<char* [S], Kokkos::LayoutRight, exec_space>;

using real_type_0d_view = typename real_type_0d_dual_view::t_dev;
using real_type_1d_view = typename real_type_1d_dual_view::t_dev;
using real_type_2d_view = typename real_type_2d_dual_view::t_dev;
using real_type_3d_view = typename real_type_3d_dual_view::t_dev;

using ordinal_type_0d_view = typename ordinal_type_0d_dual_view::t_dev;
using ordinal_type_1d_view = typename ordinal_type_1d_dual_view::t_dev;
using ordinal_type_2d_view = typename ordinal_type_2d_dual_view::t_dev;
using ordinal_type_3d_view = typename ordinal_type_3d_dual_view::t_dev;

template<int S>
using string_type_1d_view = typename string_type_1d_dual_view<S>::t_dev;

using real_type_0d_view_host = typename real_type_0d_dual_view::t_host;
using real_type_1d_view_host = typename real_type_1d_dual_view::t_host;
using real_type_2d_view_host = typename real_type_2d_dual_view::t_host;
using real_type_3d_view_host = typename real_type_3d_dual_view::t_host;

using ordinal_type_0d_view_host = typename ordinal_type_0d_dual_view::t_host;
using ordinal_type_1d_view_host = typename ordinal_type_1d_dual_view::t_host;
using ordinal_type_2d_view_host = typename ordinal_type_2d_dual_view::t_host;
using ordinal_type_3d_view_host = typename ordinal_type_3d_dual_view::t_host;

template<int S>
using string_type_1d_view_host = typename string_type_1d_dual_view<S>::t_host;

/// utility function
using do_not_init_tag = std::string; // Kokkos::ViewAllocateWithoutInitializing;
                                     // // currently not working
template<typename T>
using ats = Kokkos::ArithTraits<T>;

namespace Impl {
template<typename ViewType, typename MemoryTraitsType>
using ViewWithMemoryTraits = Kokkos::View<typename ViewType::data_type,
                                          typename ViewType::array_layout,
                                          typename ViewType::device_type,
                                          MemoryTraitsType>;

template<typename ViewType, typename MemoryTraitsType>
using ConstViewWithMemoryTraits =
  Kokkos::View<typename ViewType::const_data_type,
               typename ViewType::array_layout,
               typename ViewType::device_type,
               MemoryTraitsType>;

template<typename ViewType, typename MemoryTraitsType>
using ScratchViewWithMemoryTraits =
  Kokkos::View<typename ViewType::data_type,
               typename ViewType::array_layout,
               typename ViewType::execution_space::scratch_memory_space,
               MemoryTraitsType>;

} // namespace Impl

template<typename ViewType>
using Unmanaged =
  Impl::ViewWithMemoryTraits<ViewType, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

template<typename ViewType>
using Atomic =
  Impl::ViewWithMemoryTraits<ViewType, Kokkos::MemoryTraits<Kokkos::Atomic>>;

template<typename ViewType>
using Const =
  Impl::ConstViewWithMemoryTraits<ViewType, typename ViewType::memory_traits>;

template<typename ViewType>
using ConstUnmanaged =
  Impl::ConstViewWithMemoryTraits<ViewType,
                                  Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

template<typename ViewType>
using AtomicUnmanaged = Impl::ViewWithMemoryTraits<
  ViewType,
  Kokkos::MemoryTraits<Kokkos::Unmanaged | Kokkos::Atomic>>;

template<typename ViewType>
using Scratch =
  Impl::ScratchViewWithMemoryTraits<ViewType,
                                    Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

template<typename T>
KOKKOS_FORCEINLINE_FUNCTION T
getValueInRange(const T& lo, const T& up, const T& val)
{
  return val < lo ? lo : val > up ? up : val;
}

template<typename ViewType, typename T>
void
convertToKokkos(ViewType& out, const std::vector<T>& in)
{
#if defined(TCHEM_ENABLE_DEBUG)
  std::cout << "WARNING: we highly discourage to use this utility function and "
               "recommend to use Kokkos::View directly\n";
#endif
  {
    static_assert(Kokkos::is_view<ViewType>::value,
                  "Error: Output view is not Kokkos::View");
    static_assert(ViewType::rank == 1, "Error: Output view is not rank-1");
    static_assert(
      std::is_same<T, typename ViewType::non_const_value_type>::value,
      "Error: std::vector value type does not match to Kokkos::View value "
      "type");
    static_assert(
      std::is_same<typename ViewType::array_layout, Kokkos::LayoutRight>::value,
      "Error: Output view is supposed to be layout right");
  }

  const ordinal_type n0 = in.size(), m0 = out.extent(0);
  if (n0 != m0)
    out = ViewType(
      Kokkos::ViewAllocateWithoutInitializing("convertStdVector1D"), n0);
  const auto out_host = Kokkos::create_mirror_view(out);
  Kokkos::parallel_for(Kokkos::RangePolicy<host_exec_space>(0, n0),
                       [&](const ordinal_type& i) { out_host(i) = in[i]; });
  Kokkos::deep_copy(out, out_host);
}

template<typename ViewType, typename T>
void
convertToKokkos(ViewType& out, const std::vector<std::vector<T>>& in)
{
#if defined(TCHEM_ENABLE_DEBUG)
  std::cout << "WARNING: we highly discourage to use this utility function and "
               "recommend to use Kokkos::View directly\n";
#endif
  {
    static_assert(Kokkos::is_view<ViewType>::value,
                  "Error: Output view is not Kokkos::View");
    static_assert(ViewType::rank == 2, "Error: Output view is not rank-1");
    static_assert(
      std::is_same<T, typename ViewType::non_const_value_type>::value,
      "Error: std::vector value type does not match to Kokkos::View value "
      "type");
    static_assert(
      std::is_same<typename ViewType::array_layout, Kokkos::LayoutRight>::value,
      "Error: Output view is supposed to be layout right");
  }

  const ordinal_type m0 = out.extent(0), m1 = out.extent(1);
  const ordinal_type n0 = in.size(), n1 = in[0].size();
  if (m0 != n0 || m1 != n1)
    out = ViewType(
      Kokkos::ViewAllocateWithoutInitializing("convertStdVector2D"), n0, n1);
  const auto out_host = Kokkos::create_mirror_view(out);
  Kokkos::parallel_for(Kokkos::RangePolicy<host_exec_space>(0, n0),
                       [&](const ordinal_type& i) {
                         for (ordinal_type j = 0; j < n1; ++j)
                           out_host(i, j) = in[i][j];
                       });
  Kokkos::deep_copy(out, out_host);
}

template<typename T, typename ViewType>
void
convertToStdVector(std::vector<T>& out, const ViewType& in)
{
#if defined(TCHEM_ENABLE_DEBUG)
  std::cout << "WARNING: we highly discourage to use this utility function and "
               "recommend to use Kokkos::View directly\n";
#endif
  {
    static_assert(Kokkos::is_view<ViewType>::value,
                  "Error: Input view is not Kokkos::View");
    static_assert(ViewType::rank == 1, "Error: Input view is not rank-2");
    static_assert(
      std::is_same<T, typename ViewType::non_const_value_type>::value,
      "Error: std::vector value type does not match to Kokkos::View value "
      "type");
    static_assert(
      std::is_same<typename ViewType::array_layout, Kokkos::LayoutRight>::value,
      "Error: Input view is supposed to be layout right");
  }

  const auto in_host = Kokkos::create_mirror_view(in);
  Kokkos::deep_copy(in_host, in);

  const ordinal_type n0 = in.extent(0);
  out.resize(n0);

  Kokkos::parallel_for(Kokkos::RangePolicy<host_exec_space>(0, n0),
                       [&](const ordinal_type& i) { out[i] = in_host(i); });
}

/// latest compiler will replace the returning std::vector with move operator.
template<typename T, typename ViewType>
void
convertToStdVector(std::vector<std::vector<T>>& out, const ViewType& in)
{
#if defined(TCHEM_ENABLE_DEBUG)
  std::cout << "WARNING: we highly discourage to use this utility function and "
               "recommend to use Kokkos::View directly\n";
#endif
  {
    static_assert(Kokkos::is_view<ViewType>::value,
                  "Error: Input view is not Kokkos::View");
    static_assert(ViewType::rank == 2, "Error: Input view is not rank-2");
    static_assert(
      std::is_same<T, typename ViewType::non_const_value_type>::value,
      "Error: std::vector value type does not match to Kokkos::View value "
      "type");
    static_assert(
      std::is_same<typename ViewType::array_layout, Kokkos::LayoutRight>::value,
      "Error: Input view is supposed to be layout right");
  }

  const auto in_host = Kokkos::create_mirror_view(in);
  Kokkos::deep_copy(in_host, in);

  const ordinal_type n0 = in.extent(0);
  out.resize(n0);
  Kokkos::parallel_for(Kokkos::RangePolicy<host_exec_space>(0, n0),
                       [&](const ordinal_type& i) {
                         /// this can serialize as it uses a system function of
                         /// memory allocation but it probably does nothing
                         /// assuming that a typical use case is to receive
                         /// kokkos data which was converted from this vector)
                         const ordinal_type n1 = in.extent(1);
                         out[i].resize(n1);
                         for (ordinal_type j = 0; j < n1; ++j)
                           out[i][j] = in_host(i, j);
                       });
}

template<typename T, typename ViewType>
void
convertToStdVector(std::vector<std::vector<std::vector<T>>>& out,
                   const ViewType& in)
{
#if defined(TCHEM_ENABLE_DEBUG)
  std::cout << "WARNING: we highly discourage to use this utility function and "
               "recommend to use Kokkos::View directly\n";
#endif
  {
    static_assert(Kokkos::is_view<ViewType>::value,
                  "Error: Input view is not Kokkos::View");
    static_assert(ViewType::rank == 3, "Error: Input view is not rank-3");
    static_assert(
      std::is_same<T, typename ViewType::non_const_value_type>::value,
      "Error: std::vector value type does not match to Kokkos::View value "
      "type");
    static_assert(
      std::is_same<typename ViewType::array_layout, Kokkos::LayoutRight>::value,
      "Error: Input view is supposed to be layout right");
  }

  const auto in_host = Kokkos::create_mirror_view(in);
  Kokkos::deep_copy(in_host, in);

  const ordinal_type n0 = in.extent(0);
  out.resize(n0);
  Kokkos::parallel_for(Kokkos::RangePolicy<host_exec_space>(0, n0),
                       [&](const ordinal_type& i) {
                         /// this can serialize as it uses a system function of
                         /// memory allocation but it probably does nothing
                         /// assuming that a typical use case is to receive
                         /// kokkos data which was converted from this vector)
                         const ordinal_type n1 = in.extent(1), n2 = in.extent(2);
                         out[i].resize(n1);
                         for (ordinal_type j = 0; j < n1; ++j) {
                           out[i][j].resize(n2);
                           for (ordinal_type k = 0; k < n2; ++k)
                             out[i][j][k] = in_host(i, j, k);
                         }
                       });
}

/// Fake member type
extern Kokkos::Impl::HostThreadTeamData g_serial_thread_team_data;
struct FakeTeam
{
  static Kokkos::Impl::HostThreadTeamMember<Kokkos::Serial> getMember()
  {
    return Kokkos::Impl::HostThreadTeamMember<Kokkos::Serial>(
      *Kokkos::Impl::serial_get_thread_team_data());
  }
};

namespace Impl {
/// state vector standard interface defining the internal structure
/// 0. Density (R)
/// 1. Pressure (P)
/// 2. Temperature (T)
/// 3. MassFractions (Yn)
template<typename RealType1DView>
struct StateVector
{
private:
  const ordinal_type _nSpec;
  const RealType1DView _v;

public:
  using real_value_type = typename RealType1DView::non_const_value_type;
  using range_type = Kokkos::pair<ordinal_type, ordinal_type>;

  KOKKOS_INLINE_FUNCTION StateVector() = default;
  KOKKOS_INLINE_FUNCTION StateVector(const StateVector& b) = default;
  KOKKOS_INLINE_FUNCTION StateVector(const ordinal_type nSpec,
                                     const RealType1DView& v)
    : _nSpec(nSpec)
    , _v(v)
  {}

  /// validate input vector
  KOKKOS_INLINE_FUNCTION bool isValid() const
  {
    const bool is_valid_rank = (RealType1DView::Rank == 1);
    const bool is_extent_valid = (_v.extent(0) <= (3 + _nSpec));
    return (is_valid_rank && is_extent_valid);
  }

  /// assign a pointer to update the vector in a batch fashion
  KOKKOS_INLINE_FUNCTION ordinal_type size() const { return _nSpec + 3; }
  KOKKOS_INLINE_FUNCTION void assign_data(real_value_type* ptr)
  {
    _v.assign_data(ptr);
  }

  /// copy access to private members
  KOKKOS_INLINE_FUNCTION RealType1DView KokkosView() const { return _v; }
  KOKKOS_INLINE_FUNCTION ordinal_type NumSpecies() const { return _nSpec; }

  /// interface to state vector
  KOKKOS_INLINE_FUNCTION real_value_type& Density() const { return _v(0); }
  KOKKOS_INLINE_FUNCTION real_value_type& Pressure() const { return _v(1); }
  KOKKOS_INLINE_FUNCTION real_value_type& Temperature() const { return _v(2); }
  KOKKOS_INLINE_FUNCTION auto MassFractions() const
    -> decltype(Kokkos::subview(_v, range_type(3, 3 + _nSpec)))
  {
    return Kokkos::subview(_v, range_type(3, 3 + _nSpec));
  }

  KOKKOS_INLINE_FUNCTION real_value_type* DensityPtr() const { return &_v(0); }
  KOKKOS_INLINE_FUNCTION real_value_type* PressurePtr() const { return &_v(1); }
  KOKKOS_INLINE_FUNCTION real_value_type* TemperaturePtr() const
  {
    return &_v(2);
  }
  KOKKOS_INLINE_FUNCTION real_value_type* MassFractionsPtr() const
  {
    return &_v(3);
  }
};
static KOKKOS_INLINE_FUNCTION ordinal_type
getStateVectorSize(const ordinal_type nSpec)
{
  return nSpec + 3;
}
template<typename RealType1DView>
static KOKKOS_INLINE_FUNCTION StateVector<RealType1DView>
wrapStateVectorView(const ordinal_type nSpec, const RealType1DView& v)
{
  return StateVector<RealType1DView>(nSpec, v);
}

} // namespace Impl

///
/// These are only used by unit tests and examples
///
namespace Test {

template<typename ViewType>
static inline void
cloneView(const ViewType& v)
{
  auto vp = v.data();
  if (ViewType::Rank == 1) {
    const auto vs = v.stride(0);
    Kokkos::parallel_for(
      Kokkos::RangePolicy<typename ViewType::execution_space>(0, v.extent(0)),
      KOKKOS_LAMBDA(const ordinal_type i) { vp[i * vs] = vp[0]; });
  } else if (ViewType::Rank == 2) {
    const auto vs0 = v.stride(0);
    const auto vs1 = v.stride(1);
    Kokkos::parallel_for(
      Kokkos::RangePolicy<typename ViewType::execution_space>(0, v.extent(0)),
      KOKKOS_LAMBDA(const ordinal_type i) {
        for (ordinal_type j = 0, jend = v.extent(1); j < jend; ++j)
          vp[i * vs0 + j * vs1] = vp[j * vs1];
      });
  } else {
    std::logic_error("Error: Rank (>2) is not supported");
  }
}

template<typename RealType1DViewHostType>
static inline void
readSiteFraction(const std::string& filename,
                 const ordinal_type& nSpec,
                 const RealType1DViewHostType& siteFraction)
{
  std::ifstream file(filename);
  if (file.is_open()) {
    for (ordinal_type k = 0; k < nSpec; ++k)
      file >> siteFraction(k);
  } else {
    std::stringstream ss;
    ss << "Error: filename (" << filename << ") is not open\n";
    std::logic_error(ss.str());
  }
}

template<typename RealType1DViewHostType>
static inline void
read1DVector(const std::string& filename,
             const ordinal_type& nSpec,
             const RealType1DViewHostType& vector)
{
  std::ifstream file(filename);
  if (file.is_open()) {
    printf("read1DVector: Reading data from %s\n", filename.c_str());
    for (ordinal_type k = 0; k < nSpec; ++k)
      file >> vector(k);
  } else {
    std::stringstream ss;
    ss << "Error: filename (" << filename << ") is not open\n";
    std::logic_error(ss.str());
    printf("read1DVector: Could not open %s -> Abort !\n", filename.c_str());
    exit(1);
  }
}

template<typename StringViewHostType,
         typename RealType2DViewHostType,
         typename KCMDRealType1DViewHostType>
static inline void
readSample(const std::string& filename,
           const StringViewHostType& speciesNamesHost,
           const KCMDRealType1DViewHostType& sMass,
           const ordinal_type& nSpec,
           const ordinal_type& stateVecDim,
           RealType2DViewHostType& state_host,
           int& nBatch)
{

  //
  std::ifstream file(filename);
  std::vector<std::string> varnames;
  std::vector<real_type> values;
  if (file.is_open()) {

    printf("readSample: Reading gas samples from %s\n", filename.c_str());

    // read header of file and save variable name in vector
    std::string line;
    std::getline(file, line);
    std::istringstream iss(line);
    std::string delimiter = " ";

    size_t pos = 0;
    std::string token;
    while ((pos = line.find(delimiter)) != std::string::npos) {
      varnames.push_back(line.substr(0, pos));
      line.erase(0, pos + delimiter.length());
    }
    varnames.push_back(line);
    // for (auto i = varnames.begin(); i != varnames.end(); ++i)
    //   std::cout << *i << "\n";
    // read values
    real_type value;
    while (file >> value) {
      values.push_back(value);
    }
  } else{
    printf("readSample : Could not open %s -> Abort !\n", filename.c_str());
    exit(1);
  }

  file.close();

  std::vector<ordinal_type> indx;
  for (ordinal_type sp = 0; sp < varnames.size(); sp++) {
    for (ordinal_type i = 0; i < nSpec; i++) {
      if (strncmp(&speciesNamesHost(i, 0),
                  (varnames[sp]).c_str(),
                  LENGTHOFSPECNAME) == 0) {
        indx.push_back(i);
        printf("species %s index %d \n", &speciesNamesHost(i, 0), i);
      }
    }
  }

  nBatch = values.size() / varnames.size();
  const ordinal_type nVars = varnames.size();
  printf("Number of samples %d\n", nBatch);
  /// input: state vectors: temperature, pressure and concentration
  state_host = RealType2DViewHostType("StateVector", nBatch, stateVecDim);
  /// create a mirror view to store input from a file
  // state_host = Kokkos::create_mirror_view(state);

  Kokkos::parallel_for(Kokkos::RangePolicy<TChem::host_exec_space>(0, nBatch),
                       [state_host,nVars,&values,&indx](const ordinal_type& i) {
                         // density is set to zero

                         state_host(i, 1) = values[i * nVars + 1]; // pressure
                         state_host(i, 2) = values[i * nVars]; // temperatures

                         // mass fractions
                         for (ordinal_type sp = 0; sp < indx.size(); sp++) {
                           state_host(i, indx[sp] + 3) =
                             values[i * nVars + 2 + sp];
                         }
                       });

  //
  // 3. compute density

  Kokkos::parallel_for(
    Kokkos::RangePolicy<TChem::host_exec_space>(0, nBatch),
    [&](const ordinal_type& i) {
      const real_type_1d_view_host state_at_i =
        Kokkos::subview(state_host, i, Kokkos::ALL());
      //
      const Impl::StateVector<real_type_1d_view_host> sv_at_i(nSpec,
                                                              state_at_i);
      const auto Ys = sv_at_i.MassFractions();
      real_type Ysum(0);
      for (ordinal_type sp = 0; sp < indx.size(); sp++)
        Ysum += Ys(indx[sp]) / sMass(indx[sp]);

      const real_type Runiv = RUNIV * 1.0e3;
      sv_at_i.Density() =
        sv_at_i.Pressure() / (Runiv * Ysum * sv_at_i.Temperature());
    });
}

template<typename StringViewHostType, typename RealType2DViewHostType>
static inline void
readSurfaceSample(const std::string& filename,
                  const StringViewHostType& speciesNamesHost,
                  const ordinal_type& nSpec,
                  RealType2DViewHostType& sitefraction_host,
                  int& nBatch)
{

  //
  std::ifstream file(filename);
  std::vector<std::string> varnames;
  std::vector<real_type> values;
  if (file.is_open()) {

    printf("readSurfaceSample: Reading surface samples from %s\n", filename.c_str());

    // read header of file and save variable name in vector
    std::string line;
    std::getline(file, line);
    std::istringstream iss(line);
    std::string delimiter = " ";

    size_t pos = 0;
    std::string token;
    while ((pos = line.find(delimiter)) != std::string::npos) {
      varnames.push_back(line.substr(0, pos));
      line.erase(0, pos + delimiter.length());
    }
    varnames.push_back(line);
    // for (auto i = varnames.begin(); i != varnames.end(); ++i)
    //   std::cout << *i << "\n";
    // read values
    real_type value;
    while (file >> value) {
      values.push_back(value);
    }
  } else{
    printf("readSurfaceSample: Could not open %s -> Abort !\n", filename.c_str());
    exit(1);
  }
  file.close();

  std::vector<ordinal_type> indx;
  for (ordinal_type sp = 0; sp < varnames.size(); sp++) {
    for (ordinal_type i = 0; i < nSpec; i++) {
      if (strncmp(&speciesNamesHost(i, 0),
                  (varnames[sp]).c_str(),
                  LENGTHOFSPECNAME) == 0) {
        indx.push_back(i);
        printf("species %s index %d \n", &speciesNamesHost(i, 0), i);
      }
    }
  }

  nBatch = values.size() / varnames.size();
  const ordinal_type nVars = varnames.size();
  printf("Number of samples %d\n", nBatch);
  /// input: state vectors: temperature, pressure and concentration
  sitefraction_host =
    RealType2DViewHostType("Site fraction host", nBatch, nSpec);
  /// create a mirror view to store input from a file
  // sitefraction_host = Kokkos::create_mirror_view(state);

  Kokkos::parallel_for(Kokkos::RangePolicy<TChem::host_exec_space>(0, nBatch),
                       [sitefraction_host,nVars,&values,&indx](const ordinal_type& i) {
                         // mass fractions
                         for (ordinal_type sp = 0; sp < indx.size(); sp++) {
                           sitefraction_host(i, indx[sp]) =
                             values[i * nVars + sp];
                         }
                       });
}

template<typename StringViewHostType,
         typename RealType1DViewHostType,
         typename KCMDRealType1DViewHostType>
static inline void
readInput(const std::string& filename,
          const ordinal_type& nSpec,
          const StringViewHostType& speciesNamesHost,
          const KCMDRealType1DViewHostType& sMass,
          const RealType1DViewHostType& stateVector)
{
  // read input file that contains: T, P and some species
  // T P IC8H18 O2 N2 AR
  // compute density and set state vector.

  // 1.  get data from file
  std::ifstream file(filename);
  std::vector<std::string> varnames;
  std::vector<real_type> values;
  if (file.is_open()) {

    // read header of file and save variable name in vector
    std::string line;
    std::getline(file, line);
    std::istringstream iss(line);
    std::string delimiter = " ";

    size_t pos = 0;
    std::string token;
    while ((pos = line.find(delimiter)) != std::string::npos) {
      varnames.push_back(line.substr(0, pos));
      line.erase(0, pos + delimiter.length());
    }
    varnames.push_back(line);
    // for (auto i = varnames.begin(); i != varnames.end(); ++i)
    //         std::cout << *i << "\n";
    // read values
    real_type value;
    while (file >> value) {
      values.push_back(value);
    }
  }
  file.close();

  std::vector<ordinal_type> indx;
  for (ordinal_type sp = 0; sp < varnames.size(); sp++) {
    for (ordinal_type i = 0; i < nSpec; i++) {
      if (strncmp(&speciesNamesHost(i, 0),
                  (varnames[sp]).c_str(),
                  LENGTHOFSPECNAME) == 0) {
        indx.push_back(i);
        printf("species %s index %d \n", &speciesNamesHost(i, 0), i);
      }
    }
  }

  const ordinal_type nBatch = values.size() / varnames.size();
  const ordinal_type nVars = varnames.size();
  printf("Number of samples %d\n", nBatch);

  // 2. fill state vector with previews data
  Impl::StateVector<RealType1DViewHostType> sv(nSpec, stateVector);

  if (sv.isValid()) {
    sv.Temperature() = values[0]; // temperature
    sv.Pressure() = values[1];    // pressure
    // sv.Density()
    const auto Ys = sv.MassFractions();
    for (ordinal_type sp = 0; sp < indx.size(); sp++)
      Ys(indx[sp]) = values[sp + 2];

  } else {
    std::logic_error("Error: stateVector is not valid");
  }
  // 3. compute density
  const auto Ys = sv.MassFractions();
  real_type Ysum(0);
  for (ordinal_type sp = 0; sp < indx.size(); sp++)
    Ysum += Ys(indx[sp]) / sMass(indx[sp]);

  const real_type Runiv = RUNIV * 1.0e3;
  sv.Density() = sv.Pressure() / (Runiv * Ysum * sv.Temperature());
}

template<typename RealType1DViewHostType>
static inline void
readStateVector(const std::string& filename,
                const ordinal_type& nSpec,
                const RealType1DViewHostType& stateVector)
{
  Impl::StateVector<RealType1DViewHostType> sv(nSpec, stateVector);
  if (sv.isValid()) {
    std::ifstream file(filename);
    if (file.is_open()) {
      file >> sv.Density();
      file >> sv.Pressure();
      file >> sv.Temperature();

      const auto Xc = sv.MassFractions();
      for (ordinal_type i = 0; i < nSpec; ++i)
        file >> Xc(i);
    } else {
      std::stringstream ss;
      ss << "Error: filename (" << filename << ") is not open\n";
      std::logic_error(ss.str());
    }
  } else {
    std::logic_error("Error: stateVector is not valid");
  }
}

template<typename RealType1DViewHostType>
static inline void
writeReactionRates(const std::string& filename,
                   const ordinal_type& nSpec,
                   const RealType1DViewHostType& omega)
{
  std::ofstream file(filename);
  if (file.is_open()) {
    file << std::scientific;
    file << "## nSpec(" << nSpec << ")    Omega(" << omega.extent(0) << ")\n";
    for (ordinal_type i = 0, iend = omega.extent(0); i < iend; ++i)
      file << i << "   " << omega(i) << "\n";
  } else {
    std::stringstream ss;
    ss << "Error: filename (" << filename << ") is not open\n";
    std::logic_error(ss.str());
  }
}

template<typename RealType3DViewHostType>
static inline void
write3DMatrix(const std::string& filename, const RealType3DViewHostType& Matrix)
{
  std::ofstream file(filename);
  if (file.is_open()) {
    file << std::scientific;
    file << "## dim 1: " << Matrix.extent(0) << "\n";
    file << "## dim 2: " << Matrix.extent(1) << "\n";
    file << "## dim 3: " << Matrix.extent(2) << "\n";

    const ordinal_type len0 = Matrix.extent(0);
    const ordinal_type len1 = Matrix.extent(1);
    const ordinal_type len2 = Matrix.extent(2);
    for (ordinal_type i = 0; i < len0; ++i) {
      for (ordinal_type j = 0; j < len1; ++j) {
        for (ordinal_type k = 0; k < len2; ++k)
          file << Matrix(i, j, k) << " ";
        // file << "\n";
      }
      file << "\n";
    }
  } else {
    std::stringstream ss;
    ss << "Error: filename (" << filename << ") is not open\n";
    std::logic_error(ss.str());
  }
}

template<typename RealType1DViewHostType>
static inline void
write1DVector(const std::string& filename, const RealType1DViewHostType& Vector)
{
  std::ofstream file(filename);
  if (file.is_open()) {
    file << std::scientific;
    file << "## nitems (" << Vector.extent(0) << ")\n";

    const ordinal_type len0 = Vector.extent(0);
    for (ordinal_type i = 0; i < len0; ++i) {
      file << Vector(i) << " ";
      file << "\n";
    }
  } else {
    std::stringstream ss;
    ss << "Error: filename (" << filename << ") is not open\n";
    std::logic_error(ss.str());
  }
}

template<typename RealType2DViewHostType>
static inline void
write2DMatrix(const std::string& filename,
              const ordinal_type& nRow,
              const ordinal_type& nColumn,
              const RealType2DViewHostType& Matrix)
{
  std::ofstream file(filename);
  if (file.is_open()) {
    file << std::scientific;
    file << "## nRows(" << nRow << ")    (" << Matrix.extent(0) << ")\n";
    file << "## nCols(" << nColumn << ")    (" << Matrix.extent(1) << ")\n";

    const ordinal_type len0 = Matrix.extent(0);
    const ordinal_type len1 = Matrix.extent(1);
    for (ordinal_type i = 0; i < len0; ++i) {
      for (ordinal_type j = 0; j < len1; ++j)
        file << Matrix(i, j) << " ";
      file << "\n";
    }
  } else {
    std::stringstream ss;
    ss << "Error: filename (" << filename << ") is not open\n";
    std::logic_error(ss.str());
  }
}

template<typename RealType2DViewHostType>
static inline void
writeJacobian(const std::string& filename,
              const ordinal_type& nSpec,
              const RealType2DViewHostType& jacobian)
{
  std::ofstream file(filename);
  if (file.is_open()) {
    file << std::scientific;
    file << "## nSpec(" << nSpec << ")    R,P,T,Y_0 ... Y_{nSpec-1}("
         << jacobian.extent(0) << ")\n";

    if (jacobian.extent(0) == jacobian.extent(1)) {
      const ordinal_type len = jacobian.extent(0);
      for (ordinal_type i = 0; i < len; ++i) {
        for (ordinal_type j = 0; j < len; ++j)
          file << jacobian(i, j) << " ";
        file << "\n";
      }
    } else {
      std::logic_error("Error: jacobian is not a square matrix");
    }
  } else {
    std::stringstream ss;
    ss << "Error: filename (" << filename << ") is not open\n";
    std::logic_error(ss.str());
  }
}

static inline bool
compareFiles(const std::string& filename1, const std::string& filename2)
{
  std::ifstream f1(filename1);
  std::string s1((std::istreambuf_iterator<char>(f1)),
                 std::istreambuf_iterator<char>());
  std::ifstream f2(filename2);
  std::string s2((std::istreambuf_iterator<char>(f2)),
                 std::istreambuf_iterator<char>());
  return (s1.compare(s2) == 0);
}

} // namespace Test

} // namespace TChem

#endif
