#ifndef __TCHEM_TRANSIENT_CONT_STIRRED_TANK_REACTOR_SACADO_JACOBIAN_HPP__
#define __TCHEM_TRANSIENT_CONT_STIRRED_TANK_REACTOR_SACADO_JACOBIAN_HPP__

#include "TChem_KineticModelData.hpp"
#include "TChem_Util.hpp"
#include "TChem_Impl_TransientContStirredTankReactor_Problem.hpp"
#include "Tines.hpp"


namespace TChem {

  struct TransientContStirredTankReactorSacadoJacobian
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
      // using problem_type = TChem::Impl::TransientContStirredTankReactor_Problem<real_type, device_type>;
      // const ordinal_type m = problem_type::getNumberOfEquations(kmcd, kmcdSurf);
      const ordinal_type m = kmcd.nSpec + 1+ kmcdSurf.nSpec;
      // we need to value type to compute work size space
      ordinal_type work_size(0);
      if (m < 16) {
        using value_type = Sacado::Fad::SLFad<real_type,16>;
        work_size = Impl::TransientContStirredTankReactor_Problem<value_type, device_type>::getWorkSpaceSize(kmcd, kmcdSurf);
      } else if  (m < 32) {
        using value_type = Sacado::Fad::SLFad<real_type,32>;
        work_size = Impl::TransientContStirredTankReactor_Problem<value_type, device_type>::getWorkSpaceSize(kmcd, kmcdSurf);
      } else if  (m < 64) {
        using value_type = Sacado::Fad::SLFad<real_type,64>;
        work_size = Impl::TransientContStirredTankReactor_Problem<value_type, device_type>::getWorkSpaceSize(kmcd, kmcdSurf);
      } else if  (m < 128) {
        using value_type = Sacado::Fad::SLFad<real_type,128>;
        work_size = Impl::TransientContStirredTankReactor_Problem<value_type, device_type>::getWorkSpaceSize(kmcd, kmcdSurf);
      } else if  (m < 256) {
        using value_type = Sacado::Fad::SLFad<real_type,256>;
        work_size = Impl::TransientContStirredTankReactor_Problem<value_type, device_type>::getWorkSpaceSize(kmcd, kmcdSurf);
      } else if  (m < 512) {
        using value_type = Sacado::Fad::SLFad<real_type,512>;
        work_size = Impl::TransientContStirredTankReactor_Problem<value_type, device_type>::getWorkSpaceSize(kmcd, kmcdSurf);
      } else if (m < 1024){
        using value_type = Sacado::Fad::SLFad<real_type,1024>;
        work_size = Impl::TransientContStirredTankReactor_Problem<value_type, device_type>::getWorkSpaceSize(kmcd, kmcdSurf);
      } else{
        TCHEM_CHECK_ERROR(0,
                          "Error: Number of equations is bigger than size of sacado fad type");
      }
      return work_size + m;
    }

    static void runDeviceBatch( /// input
                               typename UseThisTeamPolicy<exec_space>::type& policy,
                               const Tines::value_type_2d_view<real_type, device_type>& state,
                               const Tines::value_type_2d_view<real_type, device_type> & site_fraction,
                               /// output
                               const Tines::value_type_3d_view<real_type, device_type>& Jacobian,
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
                             /// const data from kinetic model
                             const KineticModelConstData<host_device_type >& kmcd,
                             const KineticSurfModelConstData<host_device_type>& kmcdSurf,
                             const TransientContStirredTankReactorData<host_device_type>& cstr);

  };

} // namespace TChem

#endif
