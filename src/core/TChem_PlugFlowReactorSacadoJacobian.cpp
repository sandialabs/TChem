#include "TChem_PlugFlowReactorSacadoJacobian.hpp"

namespace TChem {
  
  /// forward decl for template function
  template<typename PolicyType,
           typename ValueType,
           typename DeviceType>
  void
  PlugFlowReactorSacadoJacobian_TemplateRun( /// input
                                            const std::string& profile_name,
                                            const ValueType& dummyValueType,
                                            /// team size setting
                                            const PolicyType& policy,

                                            const Tines::value_type_2d_view<real_type,DeviceType>& state,
                                            const Tines::value_type_2d_view<real_type, DeviceType> & site_fraction,
                                            const Tines::value_type_1d_view<real_type, DeviceType> & velocity,
                                            const Tines::value_type_3d_view<real_type,DeviceType>& Jacobian,
                                            const KineticModelConstData<DeviceType>& kmcd,
                                            const KineticSurfModelConstData<DeviceType>& kmcdSurf,
                                            const PlugFlowReactorData& pfrd);
#define TCHEM_RUN_PLUG_FLOW_REACTOR_SACADO_JACOBIAN()           \
  PlugFlowReactorSacadoJacobian_TemplateRun(                    \
                                            profile_name,       \
                                            value_type(),       \
                                            policy,             \
                                            state,              \
                                            site_fraction,      \
                                            velocity,           \
                                            Jacobian,           \
                                            kmcd,               \
                                            kmcdSurf,           \
                                            pfrd)            

  void
  PlugFlowReactorSacadoJacobian::runDeviceBatch( /// input
                                                typename UseThisTeamPolicy<exec_space>::type& policy,
                                                const Tines::value_type_2d_view<real_type, device_type>& state,
                                                const Tines::value_type_2d_view<real_type, device_type> & site_fraction,
                                                const Tines::value_type_1d_view<real_type, device_type> & velocity,
                                                /// output
                                                const Tines::value_type_3d_view<real_type, device_type>& Jacobian,
                                                /// const data from kinetic model
                                                const KineticModelConstData<device_type >& kmcd,
                                                const KineticSurfModelConstData<device_type>& kmcdSurf,
                                                const PlugFlowReactorData& pfrd)
  {
    const std::string profile_name ="TChem::PlugFlowReactorSacadoJacobian::runDeviceBatch";
    using problem_type = Impl::PlugFlowReactor_Problem<real_type, device_type>;
    const ordinal_type m = problem_type::getNumberOfEquations(kmcd, kmcdSurf);

    if (m < 16) {
      using value_type = Sacado::Fad::SLFad<real_type,16>;
      TCHEM_RUN_PLUG_FLOW_REACTOR_SACADO_JACOBIAN();
    } else if  (m < 32) {
      using value_type = Sacado::Fad::SLFad<real_type,32>;
      TCHEM_RUN_PLUG_FLOW_REACTOR_SACADO_JACOBIAN();
    } else if  (m < 64) {
      using value_type = Sacado::Fad::SLFad<real_type,64>;
      TCHEM_RUN_PLUG_FLOW_REACTOR_SACADO_JACOBIAN();
    } else if  (m < 128) {
      using value_type = Sacado::Fad::SLFad<real_type,128>;
      TCHEM_RUN_PLUG_FLOW_REACTOR_SACADO_JACOBIAN();
    } else if  (m < 256) {
      using value_type = Sacado::Fad::SLFad<real_type,256>;
      TCHEM_RUN_PLUG_FLOW_REACTOR_SACADO_JACOBIAN();
    } else if  (m < 512) {
      using value_type = Sacado::Fad::SLFad<real_type,512>;
      TCHEM_RUN_PLUG_FLOW_REACTOR_SACADO_JACOBIAN();
    } else if (m < 1024){
      using value_type = Sacado::Fad::SLFad<real_type,1024>;
      TCHEM_RUN_PLUG_FLOW_REACTOR_SACADO_JACOBIAN();
    } else{
      TCHEM_CHECK_ERROR(0,
                        "Error: Number of equations is bigger than size of sacado fad type");
    }
  }
  
  void
  PlugFlowReactorSacadoJacobian::runHostBatch( /// input
                                              typename UseThisTeamPolicy<host_exec_space>::type& policy,
                                              const Tines::value_type_2d_view<real_type, host_device_type>& state,
                                              const Tines::value_type_2d_view<real_type, host_device_type> & site_fraction,
                                              const Tines::value_type_1d_view<real_type, host_device_type> & velocity,
                                              /// output
                                              const Tines::value_type_3d_view<real_type, host_device_type>& Jacobian,
                                              /// const data from kinetic model
                                              const KineticModelConstData<host_device_type >& kmcd,
                                              const KineticSurfModelConstData<host_device_type>& kmcdSurf,
                                              const PlugFlowReactorData& pfrd)
  {
    const std::string profile_name ="TChem::PlugFlowReactorSacadoJacobian::runHostBatch";
    using problem_type = Impl::PlugFlowReactor_Problem<real_type, host_device_type>;
    const ordinal_type m = problem_type::getNumberOfEquations(kmcd, kmcdSurf);

    if (m < 16) {
      using value_type = Sacado::Fad::SLFad<real_type,16>;
      TCHEM_RUN_PLUG_FLOW_REACTOR_SACADO_JACOBIAN();
    } else if  (m < 32) {
      using value_type = Sacado::Fad::SLFad<real_type,32>;
      TCHEM_RUN_PLUG_FLOW_REACTOR_SACADO_JACOBIAN();
    } else if  (m < 64) {
      using value_type = Sacado::Fad::SLFad<real_type,64>;
      TCHEM_RUN_PLUG_FLOW_REACTOR_SACADO_JACOBIAN();
    } else if  (m < 128) {
      using value_type = Sacado::Fad::SLFad<real_type,128>;
      TCHEM_RUN_PLUG_FLOW_REACTOR_SACADO_JACOBIAN();
    } else if  (m < 256) {
      using value_type = Sacado::Fad::SLFad<real_type,256>;
      TCHEM_RUN_PLUG_FLOW_REACTOR_SACADO_JACOBIAN();
    } else if  (m < 512) {
      using value_type = Sacado::Fad::SLFad<real_type,512>;
      TCHEM_RUN_PLUG_FLOW_REACTOR_SACADO_JACOBIAN();
    } else if (m < 1024){
      using value_type = Sacado::Fad::SLFad<real_type,1024>;
      TCHEM_RUN_PLUG_FLOW_REACTOR_SACADO_JACOBIAN();
    } else{
      TCHEM_CHECK_ERROR(0,
                        "Error: Number of equations is bigger than size of sacado fad type");
    }
  }

} // namespace TChem
