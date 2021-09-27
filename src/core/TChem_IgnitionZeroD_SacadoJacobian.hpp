#ifndef __TCHEM_IGNITION_ZEROD_SACADO_JACOBIAN_HPP__
#define __TCHEM_IGNITION_ZEROD_SACADO_JACOBIAN_HPP__

#include "TChem_Util.hpp"
#include "TChem_KineticModelData.hpp"
#include "TChem_SourceTerm.hpp"

namespace TChem {

  struct IgnitionZeroD_SacadoJacobian
  {

    using host_device_type = typename Tines::UseThisDevice<host_exec_space>::type;
    using device_type      = typename Tines::UseThisDevice<exec_space>::type;

    template<typename DeviceType>
    static inline ordinal_type getWorkSpaceSize(const KineticModelConstData<DeviceType>& kmcd)
    {
      // we do not need to know value_type to compute number of equations
      using device_type = DeviceType;
      // compilation error on silver
      // using problem_type = TChem::Impl::IgnitionZeroD_Problem<real_type, device_type>;
      // const ordinal_type m = problem_type::getNumberOfEquations(kmcd);
      const ordinal_type m = kmcd.nSpec + 1;

      // we need to value type to compute work size space
      ordinal_type len(0), stw(0);
      if (m < 16) {
        using value_type = Sacado::Fad::SLFad<real_type,16>;
        len = ats<value_type>::sacadoStorageCapacity();
        stw = Impl::SourceTerm<value_type,device_type>::getWorkSpaceSize(kmcd);    
      } else if  (m < 32) {
        using value_type = Sacado::Fad::SLFad<real_type,32>;
        stw = Impl::SourceTerm<value_type,device_type>::getWorkSpaceSize(kmcd);    
        len = ats<value_type>::sacadoStorageCapacity();
      } else if  (m < 64) {
        using value_type = Sacado::Fad::SLFad<real_type,64>;
        len = ats<value_type>::sacadoStorageCapacity();
        stw = Impl::SourceTerm<value_type,device_type>::getWorkSpaceSize(kmcd);    
      } else if  (m < 128) {
        using value_type = Sacado::Fad::SLFad<real_type,128>;
        len = ats<value_type>::sacadoStorageCapacity();
        stw = Impl::SourceTerm<value_type,device_type>::getWorkSpaceSize(kmcd);    
      } else if  (m < 256) {
        using value_type = Sacado::Fad::SLFad<real_type,256>;
        len = ats<value_type>::sacadoStorageCapacity();
        stw = Impl::SourceTerm<value_type,device_type>::getWorkSpaceSize(kmcd);    
      } else if  (m < 512) {
        using value_type = Sacado::Fad::SLFad<real_type,512>;
        len = ats<value_type>::sacadoStorageCapacity();
        stw = Impl::SourceTerm<value_type,device_type>::getWorkSpaceSize(kmcd);    
      } else if (m < 1024){
        using value_type = Sacado::Fad::SLFad<real_type,1024>;
        len = ats<value_type>::sacadoStorageCapacity();
        stw = Impl::SourceTerm<value_type,device_type>::getWorkSpaceSize(kmcd);    
      } else{
        TCHEM_CHECK_ERROR(1, "Error: Number of equations is bigger than size of sacado fad type");
      }

      return stw + 2*m*len;
    }

    static void runDeviceBatch(/// input
                               typename UseThisTeamPolicy<exec_space>::type& policy,
                               const Tines::value_type_2d_view<real_type, device_type>& state,
                               /// output
                               const Tines::value_type_3d_view<real_type, device_type>& jacobian,
                               /// workspace
                               const Tines::value_type_2d_view<real_type, device_type>& workspace,
                               /// const data from kinetic model
                               const KineticModelConstData<device_type >& kmcd);

    static void runHostBatch(/// input
                             typename UseThisTeamPolicy<host_exec_space>::type& policy,
                             const Tines::value_type_2d_view<real_type, host_device_type>& state,
                             /// output
                             const Tines::value_type_3d_view<real_type, host_device_type>& jacobian,
                             /// workspace
                             const Tines::value_type_2d_view<real_type, host_device_type>& workspace,
                             /// const data from kinetic model
                             const KineticModelConstData<host_device_type >& kmcd);
  
  };

} // namespace TChem

#endif
