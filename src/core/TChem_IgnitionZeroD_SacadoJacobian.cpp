#include "TChem_IgnitionZeroD_SacadoJacobian.hpp"

namespace TChem {
  
  /// forward decl for template function
  template<typename PolicyType,
           typename ValueType,
           typename DeviceType>
  void
  IgnitionZeroD_SacadoJacobian_TemplateRun(/// input
					   const std::string& profile_name,
					   const ValueType& dummyValueType,
					   const PolicyType& policy,
					   const Tines::value_type_2d_view<real_type,DeviceType>& state,
                                           /// output
					   const Tines::value_type_3d_view<real_type,DeviceType>& jacobian,
                                           /// workspace
					   const Tines::value_type_2d_view<real_type,DeviceType>& workspace,
                                           /// kinetic model
					   const KineticModelConstData<DeviceType>& kmcd);

#define TCHEM_RUN_IGNITION_ZERO_D_SACADO_JACOBIAN()		\
  IgnitionZeroD_SacadoJacobian_TemplateRun(profile_name,	\
					   value_type(),	\
					   policy,		\
					   state,		\
					   jacobian,		\
					   workspace,		\
					   kmcd)		
  
  void
  IgnitionZeroD_SacadoJacobian::runDeviceBatch(/// input
					       typename UseThisTeamPolicy<exec_space>::type& policy,
					       const Tines::value_type_2d_view<real_type, device_type>& state,
					       /// output
					       const Tines::value_type_3d_view<real_type, device_type>& jacobian,
                                               /// workspace
					       const Tines::value_type_2d_view<real_type, device_type>& workspace,
					       /// const data from kinetic model
					       const KineticModelConstData<device_type >& kmcd)
  {
    const std::string profile_name ="TChem::IgnitionZeroD_SacadoJacobian::runDeviceBatch";
    const ordinal_type m = kmcd.nSpec + 1;

    if (m < 16) {
      using value_type = Sacado::Fad::SLFad<real_type,16>;
      TCHEM_RUN_IGNITION_ZERO_D_SACADO_JACOBIAN();
    } else if  (m < 32) {
      using value_type = Sacado::Fad::SLFad<real_type,32>;
      TCHEM_RUN_IGNITION_ZERO_D_SACADO_JACOBIAN();
    } else if  (m < 64) {
      using value_type = Sacado::Fad::SLFad<real_type,64>;
      TCHEM_RUN_IGNITION_ZERO_D_SACADO_JACOBIAN();
    } else if  (m < 128) {
      using value_type = Sacado::Fad::SLFad<real_type,128>;
      TCHEM_RUN_IGNITION_ZERO_D_SACADO_JACOBIAN();
    } else if  (m < 256) {
      using value_type = Sacado::Fad::SLFad<real_type,256>;
      TCHEM_RUN_IGNITION_ZERO_D_SACADO_JACOBIAN();
    } else if  (m < 512) {
      using value_type = Sacado::Fad::SLFad<real_type,512>;
      TCHEM_RUN_IGNITION_ZERO_D_SACADO_JACOBIAN();
    } else if (m < 1024){
      using value_type = Sacado::Fad::SLFad<real_type,1024>;
      TCHEM_RUN_IGNITION_ZERO_D_SACADO_JACOBIAN();
    } else{
      TCHEM_CHECK_ERROR(1, "Error: Number of equations is bigger than size of sacado fad type");
    }
  }

  void
  IgnitionZeroD_SacadoJacobian::runHostBatch(/// input
					     typename UseThisTeamPolicy<host_exec_space>::type& policy,
					     const Tines::value_type_2d_view<real_type, host_device_type>& state,
					     /// output
					     const Tines::value_type_3d_view<real_type, host_device_type>& jacobian,
                                             /// workspace
					     const Tines::value_type_2d_view<real_type, host_device_type>& workspace,
					     /// const data from kinetic model
					     const KineticModelConstData<host_device_type >& kmcd)
  {
    const std::string profile_name ="TChem::IgnitionZeroD_SacadoJacobian::runHostBatch";
    const ordinal_type m = kmcd.nSpec + 1;

    if (m < 16) {
      using value_type = Sacado::Fad::SLFad<real_type,16>;
      TCHEM_RUN_IGNITION_ZERO_D_SACADO_JACOBIAN();
    } else if  (m < 32) {
      using value_type = Sacado::Fad::SLFad<real_type,32>;
      TCHEM_RUN_IGNITION_ZERO_D_SACADO_JACOBIAN();
    } else if  (m < 64) {
      using value_type = Sacado::Fad::SLFad<real_type,64>;
      TCHEM_RUN_IGNITION_ZERO_D_SACADO_JACOBIAN();
    } else if  (m < 128) {
      using value_type = Sacado::Fad::SLFad<real_type,128>;
      TCHEM_RUN_IGNITION_ZERO_D_SACADO_JACOBIAN();
    } else if  (m < 256) {
      using value_type = Sacado::Fad::SLFad<real_type,256>;
      TCHEM_RUN_IGNITION_ZERO_D_SACADO_JACOBIAN();
    } else if  (m < 512) {
      using value_type = Sacado::Fad::SLFad<real_type,512>;
      TCHEM_RUN_IGNITION_ZERO_D_SACADO_JACOBIAN();
    } else if (m < 1024){
      using value_type = Sacado::Fad::SLFad<real_type,1024>;
      TCHEM_RUN_IGNITION_ZERO_D_SACADO_JACOBIAN();
    } else{
      TCHEM_CHECK_ERROR(1, "Error: Number of equations is bigger than size of sacado fad type");
    }
  }

} // namespace TChem
