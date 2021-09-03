#ifndef __TCHEM_ATMOSPHERIC_CHEMISTRY_HPP__
#define __TCHEM_ATMOSPHERIC_CHEMISTRY_HPP__

#include "TChem_KineticModelData.hpp"
#include "TChem_Util.hpp"

#include "TChem_Impl_AtmosphericChemistry.hpp"

namespace TChem {

struct AtmosphericChemistry
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
    const KineticModelNCAR_ConstData<DeviceType>& kmcd)
  {
    using device_type = DeviceType;
    using problem_type = Impl::AtmosphericChemistry_Problem<real_type, device_type>;
    const ordinal_type m = problem_type::getNumberOfEquations(kmcd);

    ordinal_type work_size(0);
#if defined(TCHEM_ENABLE_SACADO_JACOBIAN_ATMOSPHERIC_CHEMISTRY)
    if (m < 128) {
      using value_type = Sacado::Fad::SLFad<real_type,128>;
      work_size = Impl::AtmosphericChemistry<value_type, device_type>::getWorkSpaceSize(kmcd)  + m ;
    } else if  (m < 256) {
      using value_type = Sacado::Fad::SLFad<real_type,256>;
      work_size = Impl::AtmosphericChemistry<value_type, device_type>::getWorkSpaceSize(kmcd)  + m ;
    } else if  (m < 512) {
      using value_type = Sacado::Fad::SLFad<real_type,512>;
      work_size = Impl::AtmosphericChemistry<value_type, device_type>::getWorkSpaceSize(kmcd)  + m ;
    } else if (m < 1024){
      using value_type = Sacado::Fad::SLFad<real_type,1024>;
      work_size = Impl::AtmosphericChemistry<value_type, device_type>::getWorkSpaceSize(kmcd)  + m ;
    } else{
      TCHEM_CHECK_ERROR(0,
                        "Error: Number of equations is bigger than size of sacado fad type");
    }
#else
    {
      work_size = Impl::AtmosphericChemistry<real_type, device_type>::getWorkSpaceSize(kmcd)  + m ;
    }
#endif

    return work_size;

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
    const KineticModelNCAR_ConstData<device_type >& kmcd);



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
    const KineticModelNCAR_ConstData<host_device_type>& kmcd);



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
    const Kokkos::View<KineticModelNCAR_ConstData<host_device_type>*,host_device_type>& kmcds);
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
    const Kokkos::View<KineticModelNCAR_ConstData<device_type>*,device_type>& kmcds);

};

#if defined(TCHEM_ENABLE_TPL_YAML_CPP)

namespace AtmChemistry {

  template<typename StringViewHostType,
           typename RealType2DViewHostType>
  static inline void
  setScenarioConditions(const std::string& filename,
             const StringViewHostType& speciesNamesHost,
             const ordinal_type& nSpec,
             RealType2DViewHostType& state_host,
             int& nBatch)
  {

    // parser for scenario conditions
    YAML::Node root = YAML::LoadFile(filename);

    auto pressure = root["environmental_conditions"]["pressure"]["initial_value"];
    auto temperature = root["environmental_conditions"]["temperature"]["initial_value"];
    auto initial_state = root["initial_state"];
    auto species = root["species"];
    // 1. get number of batches or cells
    nBatch = pressure.size();
    printf("Number of cells %d \n",nBatch );
    // 2. find species names
    std::vector<std::string> varnames;
    for (auto const& sp_cond : initial_state) {
      auto species_name = sp_cond.first.as<std::string>();
      varnames.push_back(species_name);
    }
    // 3. get species index in TChem object
    std::vector<ordinal_type> indx;
    for (ordinal_type sp = 0; sp < varnames.size(); sp++) {
      for (ordinal_type i = 0; i < nSpec; i++) {
        if (strncmp(&speciesNamesHost(i, 0), (varnames[sp]).c_str(),
          LENGTHOFSPECNAME) == 0) {
          indx.push_back(i);
          printf("species %s index %d \n", &speciesNamesHost(i, 0), i);
          break;
        }
      }
    }

    // 3. set state vector with initial values
    state_host = real_type_2d_view_host("StateVector host", nBatch, nSpec + 3);

    const int n_spec_int = varnames.size();
    Kokkos::parallel_for(
      Kokkos::RangePolicy<host_exec_space>(0, nBatch * n_spec_int ),
      KOKKOS_LAMBDA(const int &ij) {
      const int i = ij/n_spec_int, j = ij%n_spec_int; /// m is the dimension of R
      if (j==0) {
        //3.1 only populate one time temperature and pressure
        state_host(i, 1) =  pressure[i].as<real_type>(); // pressure
        state_host(i, 2) =  temperature[i].as<real_type>(); // temperature
        // printf(" batch No %d i temp %e pressure %e \n", i, state_host(i, 2), state_host(i, 1) );
      }
      //3.2 apply unit factor
      if ( species[varnames[j]]["units"].as<std::string>() == "molecules m-3") {
            // unit conversion fractor
        const real_type conv = CONV_PPM * pressure[i].as<real_type>() / temperature[i].as<real_type>() ; //D
        state_host(i, indx[j] + 3) =  initial_state[varnames[j]][i].as<real_type>()/conv; // molar concentration

      } else if ( species[varnames[j]]["units"].as<std::string>() == "mol m-3") {
         state_host(i, indx[j] + 3) =  initial_state[varnames[j]][i].as<real_type>(); // molar concentration
      } else {
            printf("unit type does not exit %s \n", species[varnames[j]]["units"].as<std::string>().c_str());
            exit(1);
      }


      // printf(" batch No %d j %d species name %s val %e\n", i, j, varnames[j].c_str(), state_host(i, indx[j] + 3) );
    });

  }

} // namespace Atomospheric Chemistry
#endif

} // namespace TChem

#endif
