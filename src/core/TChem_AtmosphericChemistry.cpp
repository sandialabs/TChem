#include "TChem_Util.hpp"

#include "TChem_AtmosphericChemistry.hpp"

/// tadv - an input structure for time marching
/// state (nSpec+3) - initial condition of the state vector
/// qidx (lt nSpec+1) - QoI indices to store in qoi output
/// work - work space sized by getWorkSpaceSize
/// tcnt - time counter
/// qoi (time + qidx.extent(0)) - QoI output
/// kmcd - const data for kinetic model

namespace TChem {

  template<typename PolicyType,
           typename ValueType,
           typename DeviceType,
           typename TimeAdvance1DViewType>
void
AtmosphericChemistry_TemplateRunModelVariation( /// required template arguments
  const std::string& profile_name,
  const ValueType& dummyValueType,
  /// team size setting
  const PolicyType& policy,

  /// input
  const Tines::value_type_1d_view<real_type, DeviceType>& tol_newton,
  const Tines::value_type_2d_view<real_type, DeviceType>& tol_time,
  const Tines::value_type_2d_view<real_type, DeviceType>& fac,
  const TimeAdvance1DViewType& tadv,
  const Tines::value_type_2d_view<real_type, DeviceType>& state,
  /// output
  const Tines::value_type_1d_view<real_type, DeviceType>& t_out,
  const Tines::value_type_1d_view<real_type, DeviceType>& dt_out,
  const Tines::value_type_2d_view<real_type, DeviceType>& state_out,
  /// const data from kinetic model
  const Kokkos::View<KineticModelNCAR_ConstData<DeviceType >*,DeviceType>& kmcds)
{
  Kokkos::Profiling::pushRegion(profile_name);
  using policy_type = PolicyType;
  using device_type = DeviceType;

  using real_type_1d_view_type = Tines::value_type_1d_view<real_type, device_type>;
  using real_type_0d_view_type = Tines::value_type_0d_view<real_type, device_type>;

  auto kmcd_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),
						       Kokkos::subview(kmcds, 0));

  const ordinal_type level = 1;
  const ordinal_type per_team_extent = AtmosphericChemistry::getWorkSpaceSize(kmcd_host());

  Kokkos::parallel_for(
    profile_name,
    policy,
    KOKKOS_LAMBDA(const typename policy_type::member_type& member) {
      const ordinal_type i = member.league_rank();
      const auto kmcd_at_i = (kmcds.extent(0) == 1 ? kmcds(0) : kmcds(i));
      const auto tadv_at_i = tadv(i);
      const real_type t_end = tadv_at_i._tend;
      const real_type_0d_view_type t_out_at_i = Kokkos::subview(t_out, i);
      if (t_out_at_i() < t_end) {
      const real_type_1d_view_type state_at_i =
        Kokkos::subview(state, i, Kokkos::ALL());
      const real_type_1d_view_type state_out_at_i =
        Kokkos::subview(state_out, i, Kokkos::ALL());
      //
      const real_type_1d_view_type fac_at_i =
        Kokkos::subview(fac, i, Kokkos::ALL());

      const real_type_0d_view_type dt_out_at_i = Kokkos::subview(dt_out, i);
      Scratch<real_type_1d_view_type> work(member.team_scratch(level),
                                       per_team_extent);
      //
      auto wptr = work.data();
      const real_type_1d_view_type ww(wptr, work.extent(0) - (wptr - work.data()));

      Impl::StateVector<real_type_1d_view_type> sv_at_i(kmcd_at_i.nSpec, state_at_i);
      Impl::StateVector<real_type_1d_view_type> sv_out_at_i(kmcd_at_i.nSpec,
                                                        state_out_at_i);
      TCHEM_CHECK_ERROR(!sv_at_i.isValid(),
                        "Error: input state vector is not valid");
      TCHEM_CHECK_ERROR(!sv_out_at_i.isValid(),
                        "Error: input state vector is not valid");
      {
        const ordinal_type jacobian_interval =
          tadv_at_i._jacobian_interval;
        const ordinal_type max_num_newton_iterations =
          tadv_at_i._max_num_newton_iterations;
        const ordinal_type max_num_time_iterations =
          tadv_at_i._num_time_iterations_per_interval;

        const real_type dt_in = tadv_at_i._dt, dt_min = tadv_at_i._dtmin,
                        dt_max = tadv_at_i._dtmax;
	const real_type t_beg = tadv_at_i._tbeg;

        const real_type temperature = sv_at_i.Temperature();
        const real_type pressure = sv_at_i.Pressure();
        const real_type density = sv_at_i.Density();
        const real_type_1d_view_type Ys = sv_at_i.MassFractions();
        const auto activeYs = real_type_1d_view_type(Ys.data(),
                              kmcd_at_i.nSpec - kmcd_at_i.nConstSpec );
        const auto constYs  = real_type_1d_view_type(Ys.data()
                            + kmcd_at_i.nSpec - kmcd_at_i.nConstSpec,  kmcd_at_i.nSpec );

        const real_type_0d_view_type temperature_out(sv_out_at_i.TemperaturePtr());
        const real_type_0d_view_type pressure_out(sv_out_at_i.PressurePtr());
        const real_type_1d_view_type Ys_out = sv_out_at_i.MassFractions();
        const real_type_0d_view_type density_out(sv_out_at_i.DensityPtr());

        member.team_barrier();

        using atmospheric_chemistry_type =
        Impl::AtmosphericChemistry<ValueType,device_type>;

        atmospheric_chemistry_type::team_invoke(member,
                                                   jacobian_interval,
                                                   max_num_newton_iterations,
                                          max_num_time_iterations,
                                          tol_newton,
                                          tol_time,
                                          fac_at_i,
                                          dt_in,
                                          dt_min,
                                          dt_max,
                                          t_beg,
                                          t_end,
                                          temperature,
                                          pressure,
                                          constYs,
                                          activeYs,
                                          t_out_at_i,
                                          dt_out_at_i,
                                          activeYs,
                                          ww,
                                          kmcd_at_i);

        // update density and pressure with out data

        density_out() = density; // density is constant
        pressure_out()= pressure; // pressure is constant
        temperature_out() = temperature; // temperature is constant
        member.team_barrier();
      }
      }
    });
  Kokkos::Profiling::popRegion();
}

template<typename PolicyType,
         typename ValueType,
         typename DeviceType,
         typename TimeAdvance1DViewType>
void
AtmosphericChemistry_TemplateRun( /// required template arguments
  const std::string& profile_name,
  const ValueType& dummyValueType,
  /// team size setting
  const PolicyType& policy,

  /// input
  const Tines::value_type_1d_view<real_type, DeviceType>& tol_newton,
  const Tines::value_type_2d_view<real_type, DeviceType>& tol_time,
  const Tines::value_type_2d_view<real_type, DeviceType>& fac,
  const TimeAdvance1DViewType& tadv,
  const Tines::value_type_2d_view<real_type, DeviceType>& state,
  /// output
  const Tines::value_type_1d_view<real_type, DeviceType>& t_out,
  const Tines::value_type_1d_view<real_type, DeviceType>& dt_out,
  const Tines::value_type_2d_view<real_type, DeviceType>& state_out,
  /// const data from kinetic model
  const KineticModelNCAR_ConstData<DeviceType>& kmcd)
{
  Kokkos::Profiling::pushRegion(profile_name);
  using policy_type = PolicyType;
  Kokkos::View<KineticModelNCAR_ConstData<DeviceType>*,DeviceType>
    kmcds(do_not_init_tag("IgnitionaZeroD::kmcds"), 1);
  Kokkos::deep_copy(kmcds, kmcd);

  AtmosphericChemistry_TemplateRunModelVariation
    (profile_name,
     dummyValueType, policy,
     tol_newton, tol_time,
     fac,
     tadv, state,
     t_out, dt_out, state_out, kmcds);

  Kokkos::Profiling::popRegion();
}

void
AtmosphericChemistry::runDeviceBatch( /// thread block size
  typename UseThisTeamPolicy<exec_space>::type& policy,
  /// input
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
  const KineticModelNCAR_ConstData<device_type >& kmcd)
{

  #define TCHEM_RUN_ATMOSPHERIC_CHEMISTRY()                             \
          AtmosphericChemistry_TemplateRun(                                \
          profile_name,                                                   \
          value_type(),                                                   \
          policy,                                                         \
          tol_newton,                                                     \
          tol_time,                                                       \
          fac,                                                            \
          tadv,                                                           \
          state,                                                          \
          t_out,                                                          \
          dt_out,                                                         \
          state_out,                                                      \
          kmcd);                                                          \

//
 const std::string profile_name = "TChem::AtmosphericChemistry::runDeviceBatch::kmcd";

#if defined(TCHEM_ENABLE_SACADO_JACOBIAN_ATMOSPHERIC_CHEMISTRY)
 using problem_type = Impl::AtmosphericChemistry_Problem<real_type, device_type>;
 const ordinal_type m = problem_type::getNumberOfEquations(kmcd);

 if (m < 128) {
   using value_type = Sacado::Fad::SLFad<real_type,128>;
   TCHEM_RUN_ATMOSPHERIC_CHEMISTRY()
 } else if  (m < 256) {
   using value_type = Sacado::Fad::SLFad<real_type,256>;
   TCHEM_RUN_ATMOSPHERIC_CHEMISTRY()
 } else if  (m < 512) {
   using value_type = Sacado::Fad::SLFad<real_type,512>;
   TCHEM_RUN_ATMOSPHERIC_CHEMISTRY()
 } else if (m < 1024){
   using value_type = Sacado::Fad::SLFad<real_type,1024>;
   TCHEM_RUN_ATMOSPHERIC_CHEMISTRY()
 } else{
   TCHEM_CHECK_ERROR(0,
                     "Error: Number of equations is bigger than size of sacado fad type");
 }
#else
  using value_type = real_type;
  TCHEM_RUN_ATMOSPHERIC_CHEMISTRY()
#endif

}


//
void
AtmosphericChemistry::runHostBatch( /// input
  typename UseThisTeamPolicy<host_exec_space>::type& policy,
  const real_type_1d_view_host_type& tol_newton,
  const real_type_2d_view_host_type& tol_time,
  const real_type_2d_view_host_type& fac,
  const time_advance_type_1d_view_host& tadv,
  const real_type_2d_view_host_type& state,
  /// output
  const real_type_1d_view_host_type& t_out,
  const real_type_1d_view_host_type& dt_out,
  const real_type_2d_view_host_type& state_out,
  /// const data from kinetic model
  const KineticModelNCAR_ConstData<host_device_type>& kmcd)
{

//
const std::string profile_name = "TChem::AtmosphericChemistry::runHostBatch::kmcd";
#if defined(TCHEM_ENABLE_SACADO_JACOBIAN_ATMOSPHERIC_CHEMISTRY)
 using problem_type = Impl::AtmosphericChemistry_Problem<real_type, host_device_type>;
 const ordinal_type m = problem_type::getNumberOfEquations(kmcd);

 if (m < 128) {
   using value_type = Sacado::Fad::SLFad<real_type,128>;
   TCHEM_RUN_ATMOSPHERIC_CHEMISTRY()
 } else if  (m < 256) {
   using value_type = Sacado::Fad::SLFad<real_type,256>;
   TCHEM_RUN_ATMOSPHERIC_CHEMISTRY()
 } else if  (m < 512) {
   using value_type = Sacado::Fad::SLFad<real_type,512>;
   TCHEM_RUN_ATMOSPHERIC_CHEMISTRY()
 } else if (m < 1024){
   using value_type = Sacado::Fad::SLFad<real_type,1024>;
   TCHEM_RUN_ATMOSPHERIC_CHEMISTRY()
 } else{
   TCHEM_CHECK_ERROR(0,
                     "Error: Number of equations is bigger than size of sacado fad type");
 }
#else
  using value_type = real_type;
  TCHEM_RUN_ATMOSPHERIC_CHEMISTRY()
#endif
}


void
AtmosphericChemistry::runDeviceBatch( /// thread block size
  typename UseThisTeamPolicy<exec_space>::type& policy,
  /// input
  const real_type_1d_view_type& tol_newton,
  const real_type_2d_view_type& tol_time,
  const real_type_2d_view_type& fac,
  const time_advance_type_1d_view& tadv,
  const real_type_2d_view_type& state,
  /// output
  const real_type_1d_view_type& t_out,
  const real_type_1d_view_type& dt_out,
  const real_type_2d_view_type& state_out,
  /// const data from kinetic model
  const Kokkos::View<KineticModelNCAR_ConstData<device_type>*,device_type>& kmcds)
{

  #define TCHEM_RUN_ATMOSPHERIC_CHEMISTRY_MODEL_VARIATION()                             \
  AtmosphericChemistry_TemplateRunModelVariation(                          \
    profile_name,                                                         \
    value_type(),                                                         \
    policy,                                                               \
    tol_newton,                                                           \
    tol_time,                                                             \
    fac,                                                                  \
    tadv,                                                                 \
    state,                                                                \
    t_out,                                                                \
    dt_out,                                                               \
    state_out,                                                            \
    kmcds);                                                               \

    const std::string profile_name = "TChem::AtmosphericChemistry::runDeviceBatch::kmcd array";

   #if defined(TCHEM_ENABLE_SACADO_JACOBIAN_ATMOSPHERIC_CHEMISTRY)
    using problem_type = Impl::AtmosphericChemistry_Problem<real_type, device_type>;
    const ordinal_type m = problem_type::getNumberOfEquations(kmcds(0));

    if (m < 128) {
      using value_type = Sacado::Fad::SLFad<real_type,128>;
      TCHEM_RUN_ATMOSPHERIC_CHEMISTRY_MODEL_VARIATION()
    } else if  (m < 256) {
      using value_type = Sacado::Fad::SLFad<real_type,256>;
      TCHEM_RUN_ATMOSPHERIC_CHEMISTRY_MODEL_VARIATION()
    } else if  (m < 512) {
      using value_type = Sacado::Fad::SLFad<real_type,512>;
      TCHEM_RUN_ATMOSPHERIC_CHEMISTRY_MODEL_VARIATION()
    } else if (m < 1024){
      using value_type = Sacado::Fad::SLFad<real_type,1024>;
      TCHEM_RUN_ATMOSPHERIC_CHEMISTRY_MODEL_VARIATION()
    } else{
      TCHEM_CHECK_ERROR(0,
                        "Error: Number of equations is bigger than size of sacado fad type");
    }
   #else
     using value_type = real_type;
     TCHEM_RUN_ATMOSPHERIC_CHEMISTRY_MODEL_VARIATION()
   #endif

}

void
AtmosphericChemistry::runHostBatch( /// thread block size
  typename UseThisTeamPolicy<host_exec_space>::type& policy,
  /// input
  const real_type_1d_view_host_type& tol_newton,
  const real_type_2d_view_host_type& tol_time,
  const real_type_2d_view_host_type& fac,
  const time_advance_type_1d_view_host& tadv,
  const real_type_2d_view_host_type& state,
  /// output
  const real_type_1d_view_host_type& t_out,
  const real_type_1d_view_host_type& dt_out,
  const real_type_2d_view_host_type& state_out,
  /// const data from kinetic model
  const Kokkos::View<KineticModelNCAR_ConstData<host_device_type>*,host_device_type>& kmcds)
{

    const std::string profile_name = "TChem::AtmosphericChemistry::runHostBatch::kmcd array";

   #if defined(TCHEM_ENABLE_SACADO_JACOBIAN_ATMOSPHERIC_CHEMISTRY)
    using problem_type = Impl::AtmosphericChemistry_Problem<real_type, host_device_type>;
    const ordinal_type m = problem_type::getNumberOfEquations(kmcds(0));

    if (m < 128) {
      using value_type = Sacado::Fad::SLFad<real_type,128>;
      TCHEM_RUN_ATMOSPHERIC_CHEMISTRY_MODEL_VARIATION()
    } else if  (m < 256) {
      using value_type = Sacado::Fad::SLFad<real_type,256>;
      TCHEM_RUN_ATMOSPHERIC_CHEMISTRY_MODEL_VARIATION()
    } else if  (m < 512) {
      using value_type = Sacado::Fad::SLFad<real_type,512>;
      TCHEM_RUN_ATMOSPHERIC_CHEMISTRY_MODEL_VARIATION()
    } else if (m < 1024){
      using value_type = Sacado::Fad::SLFad<real_type,1024>;
      TCHEM_RUN_ATMOSPHERIC_CHEMISTRY_MODEL_VARIATION()
    } else{
      TCHEM_CHECK_ERROR(0,
                        "Error: Number of equations is bigger than size of sacado fad type");
    }
   #else
     using value_type = real_type;
     TCHEM_RUN_ATMOSPHERIC_CHEMISTRY_MODEL_VARIATION()
   #endif

}



} // namespace TChem
