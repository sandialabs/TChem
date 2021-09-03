#ifndef __TCHEM_CONST_VOLUME_IGNITION_REACTOR_HPP__
#define __TCHEM_CONST_VOLUME_IGNITION_REACTOR_HPP__

#include "TChem_KineticModelData.hpp"
#include "TChem_Util.hpp"

#include "TChem_Impl_ConstantVolumeIgnitionReactor.hpp"

namespace TChem {

struct ConstantVolumeIgnitionReactor
{

  using host_device_type = typename Tines::UseThisDevice<host_exec_space>::type;
  using device_type      = typename Tines::UseThisDevice<exec_space>::type;

  using real_type_0d_view_type = Tines::value_type_0d_view<real_type,device_type>;
  using real_type_1d_view_type = Tines::value_type_1d_view<real_type,device_type>;
  using real_type_2d_view_type = Tines::value_type_2d_view<real_type,device_type>;

  using real_type_0d_view_host_type = Tines::value_type_0d_view<real_type,host_device_type>;
  using real_type_1d_view_host_type = Tines::value_type_1d_view<real_type,host_device_type>;
  using real_type_2d_view_host_type = Tines::value_type_2d_view<real_type,host_device_type>;

  template<typename DeviceType>
  static inline ordinal_type getWorkSpaceSize(
    const KineticModelConstData<DeviceType>& kmcd)
  {
    using device_type = DeviceType;
    using problem_type = Impl::ConstantVolumeIgnitionReactor_Problem<real_type, device_type>;
    const ordinal_type m = problem_type::getNumberOfEquations(kmcd);

    ordinal_type work_size(0);
#if defined(TCHEM_ENABLE_SACADO_JACOBIAN_CONSTANT_VOLUME_IGNITION_REACTOR)
    if (m < 128) {
      using value_type = Sacado::Fad::SLFad<real_type,128>;
      work_size = Impl::ConstantVolumeIgnitionReactor<value_type, device_type>::getWorkSpaceSize(kmcd)  + m ;
    } else if  (m < 256) {
      using value_type = Sacado::Fad::SLFad<real_type,256>;
      work_size = Impl::ConstantVolumeIgnitionReactor<value_type, device_type>::getWorkSpaceSize(kmcd)  + m ;
    } else if  (m < 512) {
      using value_type = Sacado::Fad::SLFad<real_type,512>;
      work_size = Impl::ConstantVolumeIgnitionReactor<value_type, device_type>::getWorkSpaceSize(kmcd)  + m ;
    } else if (m < 1024){
      using value_type = Sacado::Fad::SLFad<real_type,1024>;
      work_size = Impl::ConstantVolumeIgnitionReactor<value_type, device_type>::getWorkSpaceSize(kmcd)  + m ;
    } else{
      TCHEM_CHECK_ERROR(0,
                        "Error: Number of equations is bigger than size of sacado fad type");
    }
#else
    {
      work_size = Impl::ConstantVolumeIgnitionReactor<real_type, device_type>::getWorkSpaceSize(kmcd)  + m ;
    }
#endif

    return work_size;

  }

  template<typename MemberType,
           typename DeviceType>
  static KOKKOS_INLINE_FUNCTION void packToValues(
    const MemberType& member,
    const real_type& temperature,
    const Tines::value_type_1d_view<real_type,DeviceType>& Ys,
    /// output
    const Tines::value_type_1d_view<real_type,DeviceType>& vals)
  {
    vals(0) = temperature;

    const ordinal_type m(Ys.extent(0));

    Kokkos::parallel_for(
      Kokkos::TeamVectorRange(member, m),
      [&](const ordinal_type& i) {vals(i + 1) = Ys(i);});
    member.team_barrier();
  }

  template<typename MemberType,
           typename DeviceType>
  static KOKKOS_INLINE_FUNCTION void unpackFromValues(
    const MemberType& member,
    const Tines::value_type_1d_view<real_type,DeviceType>& vals,
    const Tines::value_type_0d_view<real_type,DeviceType>& temperature,
    /// input
    const Tines::value_type_1d_view<real_type,DeviceType>& Ys)
  {

    temperature() = vals(0);
    const ordinal_type m(Ys.extent(0));

    Kokkos::parallel_for(Kokkos::TeamVectorRange(member, m),
                         [&](const ordinal_type& i) {
                             Ys(i) = vals(i + 1);
                         });
    member.team_barrier();
  }

  static void runDeviceBatch( /// thread block size
    typename UseThisTeamPolicy<exec_space>::type& policy,
    /// global tolerence parameters that governs all samples
    const real_type_1d_view_type& tol_newton,
    const real_type_2d_view_type& tol_time,
    /// sample specific input
    const real_type_2d_view_type& fac,
    const time_advance_type_1d_view& tadv,
    const real_type_2d_view_type& state,
    /// output
    const real_type_1d_view_type& t_out,
    const real_type_1d_view_type& dt_out,
    const real_type_2d_view_type& state_out,
    /// const data from kinetic model
    const KineticModelConstData<device_type >& kmcd);



  /// tadv - an input structure for time marching
  /// state (nSpec+3) - initial condition of the state vector
  /// work - work space sized by getWorkSpaceSize
  /// t_out - time when this code exits
  /// state_out - final condition of the state vector (the same input state can
  /// be overwritten) kmcd - const data for kinetic model
  static void runHostBatch( /// input
    typename UseThisTeamPolicy<host_exec_space>::type& policy,
    /// global tolerence parameters that governs all samples
    const real_type_1d_view_host_type& tol_newton,
    const real_type_2d_view_host_type& tol_time,
    /// sample specific input
    const real_type_2d_view_host_type& fac,
    const time_advance_type_1d_view_host& tadv,
    const real_type_2d_view_host_type& state,
    /// output
    const real_type_1d_view_host_type& t_out,
    const real_type_1d_view_host_type& dt_out,
    const real_type_2d_view_host_type& state_out,
    /// const data from kinetic model
    const KineticModelConstData<host_device_type>& kmcd);



  static void runHostBatch( /// input
    typename UseThisTeamPolicy<host_exec_space>::type& policy,
    /// global tolerence parameters that governs all samples
    const real_type_1d_view_host_type& tol_newton,
    const real_type_2d_view_host_type& tol_time,
    /// sample specific input
    const real_type_2d_view_host_type& fac,
    const time_advance_type_1d_view_host& tadv,
    const real_type_2d_view_host_type& state,
    /// output
    const real_type_1d_view_host_type& t_out,
    const real_type_1d_view_host_type& dt_out,
    const real_type_2d_view_host_type& state_out,
    /// const data from kinetic model
    const Kokkos::View<KineticModelConstData<host_device_type>*,host_device_type>& kmcds);
  //
  static void runDeviceBatch( /// thread block size
    typename UseThisTeamPolicy<exec_space>::type& policy,
    /// global tolerence parameters that governs all samples
    const real_type_1d_view_type& tol_newton,
    const real_type_2d_view_type& tol_time,
    /// sample specific input
    const real_type_2d_view_type& fac,
    const time_advance_type_1d_view& tadv,
    const real_type_2d_view_type& state,
    /// output
    const real_type_1d_view_type& t_out,
    const real_type_1d_view_type& dt_out,
    const real_type_2d_view_type& state_out,
    /// const data from kinetic model
    const Kokkos::View<KineticModelConstData<device_type>*,device_type>& kmcds);

};

} // namespace TChem

#endif
