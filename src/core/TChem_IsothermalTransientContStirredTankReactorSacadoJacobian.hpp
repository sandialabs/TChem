#ifndef __TCHEM_ISOTHERMAL_TRANSIENT_CONT_STIRRED_TANK_REACTOR_SACADO_JACOBIAN_HPP__
#define __TCHEM_ISOTHERMAL_TRANSIENT_CONT_STIRRED_TANK_REACTOR_SACADO_JACOBIAN_HPP__

#include "TChem_KineticModelData.hpp"
#include "TChem_Util.hpp"
#include "TChem_Impl_IsothermalTransientContStirredTankReactor_Problem.hpp"
#include "TChem_Impl_IsothermalTransientContStirredTankReactorRHS.hpp"
#include "Tines.hpp"


namespace TChem {

  struct IsothermalTransientContStirredTankReactorSacadoJacobian
  {

    using host_device_type = typename Tines::UseThisDevice<host_exec_space>::type;
    using device_type      = typename Tines::UseThisDevice<exec_space>::type;


    template<typename DeviceType>
    static inline ordinal_type getWorkSpaceSize(
                                                const KineticModelConstData<DeviceType>& kmcd,
                                                const KineticSurfModelConstData<DeviceType>& kmcdSurf )
    {

      // we do not need to know value_type to compute number of equations
      using device_type = DeviceType;
      // compilation error on silver
      // using problem_type = TChem::Impl::IsothermalTransientContStirredTankReactor_Problem<real_type, device_type>;
      // const ordinal_type m = problem_type::getNumberOfEquations(kmcd, kmcdSurf);
      const ordinal_type m = kmcd.nSpec + kmcdSurf.nSpec;
      const ordinal_type src_workspace_size = Impl::IsothermalTransientContStirredTankReactorRHS<real_type, device_type>
      ::getWorkSpaceSize(kmcd, kmcdSurf);
      // we need to value type to compute work size space
      const ordinal_type workscape_src_jacobian = 2 * m + src_workspace_size;
      ordinal_type sacadoStorageCapacity(1);
      if (m < 16) {
        using value_type = Sacado::Fad::SLFad<real_type,16>;
        sacadoStorageCapacity = ats<value_type>::sacadoStorageCapacity();
      } else if  (m < 32) {
        using value_type = Sacado::Fad::SLFad<real_type,32>;
        sacadoStorageCapacity = ats<value_type>::sacadoStorageCapacity();
      } else if  (m < 64) {
        using value_type = Sacado::Fad::SLFad<real_type,64>;
        sacadoStorageCapacity = ats<value_type>::sacadoStorageCapacity();
      } else if  (m < 128) {
        using value_type = Sacado::Fad::SLFad<real_type,128>;
        sacadoStorageCapacity = ats<value_type>::sacadoStorageCapacity();
      } else if  (m < 256) {
        using value_type = Sacado::Fad::SLFad<real_type,256>;
        sacadoStorageCapacity = ats<value_type>::sacadoStorageCapacity();
      } else if  (m < 512) {
        using value_type = Sacado::Fad::SLFad<real_type,512>;
        sacadoStorageCapacity = ats<value_type>::sacadoStorageCapacity();
      } else if (m < 1024){
        using value_type = Sacado::Fad::SLFad<real_type,1024>;
        sacadoStorageCapacity = ats<value_type>::sacadoStorageCapacity();
      } else{
        TCHEM_CHECK_ERROR(0,
                          "Error: Number of equations is bigger than size of sacado fad type");
      }
      return workscape_src_jacobian * sacadoStorageCapacity + m;
    }

    static void runDeviceBatch( /// input
                               typename UseThisTeamPolicy<exec_space>::type& policy,
                               const Tines::value_type_2d_view<real_type, device_type>& state,
                               const Tines::value_type_2d_view<real_type, device_type> & site_fraction,
                               /// output
                               const Tines::value_type_3d_view<real_type, device_type>& Jacobian,
                               const Tines::value_type_2d_view<real_type, device_type> & rhs,
                               /// const data from kinetic model
                               const KineticModelConstData<device_type >& kmcd,
                               const KineticSurfModelConstData<device_type>& kmcdSurf,
                               const TransientContStirredTankReactorData<device_type>& cstr);
    //
    static void runHostBatch( /// input
                             typename UseThisTeamPolicy<host_exec_space>::type& policy,
                             const Tines::value_type_2d_view<real_type, host_device_type>& state,
                             const Tines::value_type_2d_view<real_type, host_device_type> & site_fraction,
                             /// output
                             const Tines::value_type_3d_view<real_type, host_device_type>& Jacobian,
                             const Tines::value_type_2d_view<real_type, host_device_type> & rhs,
                             /// const data from kinetic model
                             const KineticModelConstData<host_device_type >& kmcd,
                             const KineticSurfModelConstData<host_device_type>& kmcdSurf,
                             const TransientContStirredTankReactorData<host_device_type>& cstr);

  };

} // namespace TChem

#endif
