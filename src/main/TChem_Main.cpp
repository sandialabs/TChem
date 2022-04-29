// clang-format off
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
// clang-format on

#include "TChem.hpp"
#include "TChem_Main_Util.hpp"
#include "TChem_CommandLineParser.hpp"

int main(int argc, char *argv[]) {

  std::string json_filename("input.json");
  int verbose(1);
  TChem::CommandLineParser opts("tchem.x");
  opts.set_option<std::string>("input", "input json filename", &json_filename);
  opts.set_option<int>("verbose", "verbose output level 0) none ... 4) detailed verbose", &verbose);

  ///
  /// command line input parser
  ///
  {
    const bool r_parse = opts.parse(argc, argv);
    if (r_parse)
      return 0; // print help return
  }

  ///
  /// input json file parser
  ///
  boost::property_tree::ptree tree;
  try {
    boost::json::value jin;
    const int r_val = parse_json(json_filename, jin);
    TCHEM_CHECK_ERROR(r_val, "Error: fails to parse JSON");

    std::stringstream ss;
    ss << jin;
    boost::property_tree::json_parser::read_json(ss, tree);
  } catch (const std::exception &e) {
    std::cerr << "Error: exception is caught converting property tree\n" << e.what() << "\n";
  }

  if (verbose > 3) {
    std::cout << "-- Input JSON file after sanitizing\n";
    boost::property_tree::json_parser::write_json(std::cout, tree);
  }

  ///
  /// validate property tree
  ///
  boost::property_tree::ptree root;
  try {
    const int r_val = validate(tree, root);
    TCHEM_CHECK_ERROR(r_val, "Error: fails to validate input property tree");
  } catch (const std::exception &e) {
    std::cerr << "Error: exception is caught validating property tree input\n" << e.what() << "\n";
    return -1;
  }

  using ordinal_type = TChem::ordinal_type;
  using real_type = TChem::real_type;
  using time_advance_type = TChem::time_advance_type;

  using real_type_0d_view = TChem::real_type_0d_view;
  using real_type_1d_view = TChem::real_type_1d_view;
  using real_type_2d_view = TChem::real_type_2d_view;

  using time_advance_type_0d_view = TChem::time_advance_type_0d_view;
  using time_advance_type_1d_view = TChem::time_advance_type_1d_view;

  using real_type_0d_view_host = TChem::real_type_0d_view_host;
  using real_type_1d_view_host = TChem::real_type_1d_view_host;
  using real_type_2d_view_host = TChem::real_type_2d_view_host;

  using time_advance_type_0d_view_host = TChem::time_advance_type_0d_view_host;
  using time_advance_type_1d_view_host = TChem::time_advance_type_1d_view_host;

  ///
  /// run the code
  ///
  Kokkos::initialize(argc, argv);
  {
    using exec_space = TChem::exec_space;
    using host_exec_space = TChem::host_exec_space;

    if (verbose > 0) {
#if defined(KOKKOS_ENABLE_SERIAL)
      if (std::is_same<host_exec_space, Kokkos::Serial>::value)
        std::cout << "Host: Serial\n";
      if (std::is_same<exec_space, Kokkos::Serial>::value)
        std::cout << "Device: Serial\n";
#endif
#if defined(KOKKOS_ENABLE_OPENMP)
      if (std::is_same<host_exec_space, Kokkos::OpenMP>::value)
        std::cout << "Host: OpenMP\n";
      if (std::is_same<exec_space, Kokkos::OpenMP>::value)
        std::cout << "Device: OpenMP\n";
      Kokkos::OpenMP().print_configuration(std::cout, false);
#endif
#if defined(KOKKOS_ENABLE_CUDA)
      if (std::is_same<exec_space, Kokkos::Cuda>::value)
        std::cout << "Device: Cuda\n";
      Kokkos::Cuda().print_configuration(std::cout, false);
#endif
    }

    using device_type = typename Tines::UseThisDevice<exec_space>::type;
    using host_device_type = typename Tines::UseThisDevice<host_exec_space>::type;

    ///
    /// # of samples
    ///
    /// when input file has zero, it will be re-defined as the number of samples in the input file
    ///
    ordinal_type n_samples(0);
    {
      std::string string_n_samples;
      if (validate_optional(root, "number of samples", string_n_samples)) {
        if (string_n_samples == "auto")
          n_samples = 0;
        else
          n_samples = stoi(string_n_samples);
      }
    }

    ///
    /// run parameters
    ///
    ordinal_type team_size(-1), vector_size(-1);
    try {
      boost::property_tree::ptree node;
      if (validate_optional(root, "run", node)) {
        std::string string_team_size;
        if (validate_optional(node, "team size", string_team_size)) {
          if (string_team_size == "auto") {
            team_size = -1;
          } else {
            team_size = stoi(string_team_size);
          }
        }
        std::string string_vector_size;
        if (validate_optional(node, "vector size", string_vector_size)) {
          if (string_vector_size == "auto") {
            vector_size = -1;
          } else {
            vector_size = stoi(string_vector_size);
          }
        }
      }
    } catch (const std::exception &e) {
      std::cerr << "Error: exception is caught parsing run parameters\n" << e.what() << "\n";
    }

    ///
    /// construct kintic model
    ///
    TChem::KineticModelData kmd;
    try {
      boost::property_tree::ptree node;
      validate_required(root, "gas kinetic model", node);
      std::string kmd_type_name;
      validate_required(node, "type", kmd_type_name);

      std::string chem_file_name;
      {
        std::string filename;
        validate_required(node, "input file name", filename);
        chem_file_name = replace_env_variable(filename);
      }
      if (kmd_type_name == "chemkin") {
        std::string thermo_file_name;
        {
          std::string filename;
          validate_required(node, "thermo file name", filename);
          thermo_file_name = replace_env_variable(filename);
        }
        kmd = TChem::KineticModelData(chem_file_name, thermo_file_name);
      } else if (kmd_type_name == "cantera-yaml") {
        kmd = TChem::KineticModelData(chem_file_name);
      }
    } catch (const std::exception &e) {
      std::cerr << "Error: exception is caught constructing a kinetic model\n" << e.what() << "\n";
    }

    const TChem::KineticModelConstData<device_type> kmcd = TChem::createGasKineticModelConstData<device_type>(kmd);
    const auto kmcd_host = TChem::createGasKineticModelConstData<host_device_type>(kmd);

    if (verbose > 0) {
      std::cout << "# of species: " << kmcd_host.nSpec << "\n";
      std::cout << "# of reactions: " << kmcd_host.nReac << "\n";
    }

    ///
    /// create input from file
    ///
    std::vector<std::string> var_name;
    std::vector<std::vector<real_type>> init_state_vector;
    try {
      boost::property_tree::ptree node;
      validate_required(root, "input", node);
      validate_required(node, "state vector", node);
      std::string input_filename;
      {
        std::string filename;
        validate_required(node, "file name", filename);
        input_filename = replace_env_variable(filename);
      }

      boost::json::value jin;
      const int r_val = parse_json(input_filename, jin);
      TCHEM_CHECK_ERROR(r_val, "Error: fails to parse JSON");

      var_name = boost::json::value_to<std::vector<std::string>>(jin.at("variable name"));
      init_state_vector = boost::json::value_to<std::vector<std::vector<real_type>>>(jin.at("state vector"));

      if (n_samples == 0) {
        n_samples = init_state_vector.size();
      } else {
        std::cout << "n samples = " << n_samples << " , " << init_state_vector.size() << "\n";
        TCHEM_CHECK_ERROR(
            n_samples > init_state_vector.size(),
            "Error: n_samples is non zero value and input state vector has a smaller number of samples than n_samples");
        for (ordinal_type i = 0; i < n_samples; ++i) {
          TCHEM_CHECK_ERROR(var_name.size() != init_state_vector.at(i).size(),
                            "Error: input state vector does not match to the header description");
        }
      }
    } catch (const std::exception &e) {
      std::cerr << "Error: exception is caught reading input state vector\n" << e.what() << "\n";
    }

    ///
    /// allocate state vector and assign the input state vector
    ///

    /// create index from list of species names
    std::vector<ordinal_type> var_index;
    var_index.resize(var_name.size());
    try {
      const auto species_name = kmcd_host.speciesNames;
      TCHEM_CHECK_ERROR(var_name.size() > (species_name.extent(0) + 3),
                        "Error: the number of input variables is bigger than species in kinetic model");

      if (verbose > 0) {
        std::cout << "# of species in input: " << var_name.size() << "\n";
        std::cout << "# of species in kinetic model: " << species_name.extent(0) << "\n";
        std::cout << "  [";
        for (ordinal_type j = 0, jend = species_name.extent(0); j < jend; ++j) {
          std::cout << std::string(&species_name(j, 0)) << ",";
        }
        std::cout << "]\n";
      }

      for (ordinal_type i = 0, iend = var_name.size(); i < iend; ++i) {
        bool found(false);
        for (ordinal_type j = 0, jend = species_name.extent(0); j < jend; ++j) {
          if (strncmp(var_name.at(i).c_str(), &species_name(j, 0), LENGTHOFSPECNAME) == 0) {
            var_index.at(i) = j + 3;
            found = true;
          } else if (var_name.at(i) == "R") {
            var_index.at(i) = 0;
            found = true;
          } else if (var_name.at(i) == "P") {
            var_index.at(i) = 1;
            found = true;
          } else if (var_name.at(i) == "T") {
            var_index.at(i) = 2;
            found = true;
          }
          if (found) {
            if (verbose > 0) {
              std::cout << i << " species: " << var_name.at(i) << ", index: " << var_index.at(i) << "\n";
            }
            break;
          }
        }
        {
          std::stringstream ss;
          ss << "Error: variable (" << var_name.at(i) << ") does not exist in kinetic model\n";
          TCHEM_CHECK_ERROR(!found, ss.str().c_str());
        }
      }
    } catch (const std::exception &e) {
      std::cerr << "Error: exception is caught reading input state vector\n" << e.what() << "\n";
    }

    /// remap
    const ordinal_type sv_size = TChem::Impl::getStateVectorSize(kmcd_host.nSpec);
    real_type_2d_view sv("state vector", n_samples, sv_size);
    auto sv_host = Kokkos::create_mirror_view(sv);
    if (verbose > 0) {
      std::cout << "state vector size used in tchem (rho, p, t, Yn): " << sv_size << "\n";
    }

    try {
      const auto s_mass_host = kmcd_host.sMass;
      Kokkos::parallel_for(Kokkos::RangePolicy<host_exec_space>(0, n_samples), [&](const ordinal_type i) {
        /// p, t, Yn is set
        real_type ysum(0);
        for (ordinal_type j = 0, jend = var_index.size(); j < jend; ++j) {
          const ordinal_type jj = var_index.at(j), sp = jj - 3;
          sv_host(i, jj) = init_state_vector.at(i).at(j);

          /// if species index is valid, contribute to ysum
          if (sp >= 0) {
            ysum += sv_host(i, jj) / s_mass_host(sp);
          }
        }

        /// density sv(i,0), pressure sv(i,1), temperature sv(i,2)
        sv_host(i, 0) = sv_host(i, 1) / (RUNIV * 1.0e3 * ysum * sv_host(i, 2));
      });
      Kokkos::deep_copy(sv, sv_host);
    } catch (const std::exception &e) {
      std::cerr << "Error: exception is caught remapping state vector\n" << e.what() << "\n";
    }

    ///
    /// reactor setup
    ///
    std::string reactor_type_name;
    ordinal_type n_equations(0);
    try {
      boost::property_tree::ptree node;
      // To DO
      /// KK: this is no longer required since we have function evaluation output
      ///     change required to optional; we do not do both reactor and function evaluations
      validate_required(root, "reactor", node);
      validate_required(node, "type", reactor_type_name);

      if (reactor_type_name == "constant volume homogeneous batch reactor") {
        using problem_type = TChem::Impl::ConstantVolumeIgnitionReactor_Problem<real_type, host_device_type>;
        n_equations = problem_type::getNumberOfTimeODEs(kmcd_host);
      } else if (reactor_type_name == "constant pressure homogeneous batch reactor") {
        using problem_type = TChem::Impl::IgnitionZeroD_Problem<real_type, host_device_type>;
        n_equations = problem_type::getNumberOfTimeODEs(kmcd_host);
      } else {
        std::string msg("Error: not yet implemented: " + reactor_type_name);
        TCHEM_CHECK_ERROR(true, msg.c_str());
      }
    } catch (const std::exception &e) {
      std::cerr << "Error: exception is caught parsing reactor\n" << e.what() << "\n";
    }

    ///
    /// kintic model array (soft copy)
    ///
    TChem::kmd_type_1d_view_host kmds;
    try {
      kmds = kmd.clone(n_samples);
    } catch (const std::exception &e) {
      std::cerr << "Error: exception is caught creating clones of kinetic model\n" << e.what() << "\n";
    }

    ///
    /// model variation of pre-exponential factors
    ///
    try {
      boost::property_tree::ptree node;
      if (validate_optional(root, "gas model variation", node)) {
        std::string input_filename;
        {
          std::string filename;
          validate_required(node, "file name", filename);
          input_filename = replace_env_variable(filename);
        }

        boost::json::value jin;
        const int r_val = parse_json(input_filename, jin);
        TCHEM_CHECK_ERROR(r_val, "Error: fails to parse JSON");

        std::vector<std::vector<real_type>> pre_exponential_factor;
        std::vector<ordinal_type> reaction_index;

        reaction_index = boost::json::value_to<std::vector<ordinal_type>>(jin.at("reaction_index"));
        pre_exponential_factor =
            boost::json::value_to<std::vector<std::vector<real_type>>>(jin.at("pre_exponential_factor"));
        {
          std::stringstream ss;
          ss << "Error: n_samples (" << n_samples << ") does not match to model variation input ("
             << pre_exponential_factor.size() << ")\n";
          TCHEM_CHECK_ERROR(n_samples != pre_exponential_factor.size(), ss.str().c_str());
        }

        const ordinal_type n_reactions = reaction_index.size();
        {
          std::stringstream ss;
          ss << "Error: number of modified reactions (" << n_reactions
             << ") does not match to pre-exponential factor size (" << pre_exponential_factor[0].size() << ")\n";
          TCHEM_CHECK_ERROR(n_reactions != pre_exponential_factor[0].size(), ss.str().c_str());
        }

        if (verbose > 0) {
          std::cout << "number of reactions being modified: " << n_reactions << "\n";
        }

        real_type_3d_view_host factors("factors", n_samples, n_reactions, 3);
        Kokkos::deep_copy(factors, real_type(1));

        auto pre_exp_factor = Kokkos::subview(factors, Kokkos::ALL(), Kokkos::ALL(), 0);
        // auto act_energy_factor = Kokkos::subview(factors, Kokkos::ALL(), Kokkos::ALL(), 1);
        // auto temp_coef_factor = Kokkos::subview(factors, Kokkos::ALL(), Kokkos::ALL(), 2);

        Kokkos::parallel_for(Kokkos::RangePolicy<host_exec_space>(0, n_samples * n_reactions),
                             [&](const ordinal_type &ij) {
                               const ordinal_type i = ij / n_reactions, j = ij % n_reactions;
                               pre_exp_factor(i, j) = pre_exponential_factor.at(i).at(j);
                             });
        constexpr ordinal_type gas(0);
        const ordinal_type_1d_view_host reaction_index_view(reaction_index.data(), n_reactions);
        TChem::modifyArrheniusForwardParameter(kmds, gas, reaction_index_view, factors);
      }
    } catch (const std::exception &e) {
      std::cerr << "Error: exception is caught reading model variation \n" << e.what() << " \n";
    }

    ///
    /// sensitivity
    ///
    std::string sensitivity_type_name;
    bool solve_tla(false);
    real_type theta(0.5);
    real_type_3d_view sz;
    try {
      boost::property_tree::ptree node;
      if (validate_optional(root, "sensitivity to ignition delay time", node)) {
        std::string sensitivity_type_name;
        validate_required(node, "type", sensitivity_type_name);
        if (sensitivity_type_name == "disabled") {
          solve_tla = false;
        } else {
          solve_tla = true;
          validate_optional(node, "theta", theta);

          /// TODO:: this allocation should follow the sensitivity source term
          sz = real_type_3d_view("state z", n_samples, kmcd_host.nSpec + 1, kmcd_host.nReac);
        }
      }
    } catch (const std::exception &e) {
      std::cerr << "Error: exception is caught allocating state z\n" << e.what() << "\n";
    }
    auto sz_host = Kokkos::create_mirror_view(sz);

    ///
    /// time integrator
    ///
    std::string time_integrator_type_name;

    ordinal_type max_num_kernel_launch(0);
    time_advance_type tadv_default;
    real_type_2d_view tol_time("tol time", n_equations, 2);
    real_type_1d_view tol_newton("tol newton", 2);

    auto tol_time_host = Kokkos::create_mirror_view(tol_time);
    auto tol_newton_host = Kokkos::create_mirror_view(tol_newton);

    try {
      boost::property_tree::ptree time_integrator_node;
      validate_required(root, "time integrator", time_integrator_node);

      std::string time_integrator_type_name;
      validate_required(time_integrator_node, "type", time_integrator_type_name);
      {
        boost::property_tree::ptree node;
        validate_required(time_integrator_node, "newton solver", node);
        validate_required(node, "absolute tolerance", tol_newton_host(0));
        validate_required(node, "relative tolerance", tol_newton_host(1));

        tadv_default._jacobian_interval = 5;
        validate_optional(node, "jacobian evaluation interval", tadv_default._jacobian_interval);

        tadv_default._max_num_newton_iterations = 20;
        validate_optional(node, "max number of newton iterations", tadv_default._max_num_newton_iterations);
      }
      {
        boost::property_tree::ptree node;
        validate_required(time_integrator_node, "time", node);

        tadv_default._tbeg = 0;
        validate_optional(node, "time begin", tadv_default._tbeg);
        validate_required(node, "time end", tadv_default._tend);

        validate_required(node, "min time step size", tadv_default._dtmin);
        validate_required(node, "max time step size", tadv_default._dtmax);

        tadv_default._dt = tadv_default._dtmin;

        real_type atol_time_default, rtol_time_default;
        validate_required(node, "absolute tolerance", atol_time_default);
        validate_required(node, "relative tolerance", rtol_time_default);
        for (ordinal_type i = 0; i < n_equations; ++i) {
          tol_time_host(i, 0) = atol_time_default;
          tol_time_host(i, 1) = rtol_time_default;
        }

        max_num_kernel_launch = 100000;
        validate_optional(node, "max number of kernel launch", max_num_kernel_launch);

        tadv_default._num_time_iterations_per_interval = 100;
        validate_optional(node, "number of internal time iterations", tadv_default._num_time_iterations_per_interval);

        tadv_default._num_outer_time_iterations_per_interval = 1;
        validate_optional(node, "number of outer time iterations",
                          tadv_default._num_outer_time_iterations_per_interval);
      }

      Kokkos::deep_copy(tol_time, tol_time_host);
      Kokkos::deep_copy(tol_newton, tol_newton_host);
    } catch (const std::exception &e) {
      std::cerr << "Error: exception is caught setting time integrator\n" << e.what() << "\n";
    }

    time_advance_type_1d_view tadv("tadv", n_samples);
    Kokkos::deep_copy(tadv, tadv_default);

    ///
    /// output
    ///
    std::string output_state_vector_filename;
    std::string output_ignition_delay_time_filename;
    std::string output_sensitivity_to_ignition_delay_time_filename;
    real_type temperature_threshold_to_ignition_delay_time;

    std::ofstream ofs_sv, ofs_idt, ofs_s2idt;
    bool is_sv_write(false), is_idt_write(false), is_s2idt_write(false);
    try {
      boost::property_tree::ptree output_node;
      if (validate_optional(root, "output", output_node)) {
        {
          boost::property_tree::ptree node;
          if (validate_optional(output_node, "state vector", node)) {
            std::string filename;
            validate_required(node, "file name", filename);
            output_state_vector_filename = replace_env_variable(filename);

            ofs_sv.open(output_state_vector_filename);
            {
              std::string msg("Error: fail to open file " + output_state_vector_filename);
              TCHEM_CHECK_ERROR(!ofs_sv.is_open(), msg.c_str());
            }
            is_sv_write = true;
          }
        }
        {
          boost::property_tree::ptree node;
          if (validate_optional(output_node, "ignition delay time", node)) {
            validate_required(node, "threshold temperature", temperature_threshold_to_ignition_delay_time);
            std::string filename;
            validate_required(node, "file name", filename);
            output_ignition_delay_time_filename = replace_env_variable(filename);

            ofs_idt.open(output_ignition_delay_time_filename);
            {
              std::string msg("Error: fail to open file " + output_ignition_delay_time_filename);
              TCHEM_CHECK_ERROR(!ofs_idt.is_open(), msg.c_str());
            }
            is_idt_write = true;
          }
        }
        {
          boost::property_tree::ptree node;
          if (validate_optional(output_node, "sensitivity to ignition delay time", node)) {
            std::string filename;
            validate_required(node, "file name", filename);
            output_sensitivity_to_ignition_delay_time_filename = replace_env_variable(filename);

            ofs_s2idt.open(output_sensitivity_to_ignition_delay_time_filename);
            {
              std::string msg("Error: fail to open file " + output_sensitivity_to_ignition_delay_time_filename);
              TCHEM_CHECK_ERROR(!ofs_s2idt.is_open(), msg.c_str());
            }
            is_s2idt_write = true;
          }
        }
      }
    } catch (const std::exception &e) {
      std::cerr << "Error: exception is caught setting time integrator\n" << e.what() << "\n";
    }

    /// header
    std::string indent("    "), indent2(indent + indent), indent3(indent2 + indent);

    /// state vectors
    if (is_sv_write) {
      ofs_sv << "{\n" << indent << "\"variable name\" : [ \"density\",\"pressure\",\"temperature\"";
      const auto species_name = kmcd_host.speciesNames;
      for (ordinal_type i = 0, iend = species_name.extent(0); i < iend; ++i) {
        ofs_sv << ",\"" << std::string(&species_name(i, 0)) << "\"";
      }
      ofs_sv << "],\n";
      ofs_sv << indent << "\"number of samples\" :" << n_samples << ",\n";
      ofs_sv << indent << "\"state vector\" : [\n";
    }
    auto write_sv = [&ofs_sv, is_sv_write, indent2, indent3](const ordinal_type s, const real_type t,
                                                             const real_type dt, const real_type_1d_view_host &data) {
      if (is_sv_write) {
        ofs_sv << std::scientific << std::setprecision(15);
        ofs_sv << indent2 << "{\n";
        ofs_sv << indent3 << "\"sample\" : " << s << ",\n";
        ofs_sv << indent3 << "\"t\" : " << t << ",\n";
        ofs_sv << indent3 << "\"dt\" : " << dt << ",\n";
        ofs_sv << indent3 << "\"data\" : [ ";
        ofs_sv << data(0);
        for (ordinal_type i = 1, iend = data.extent(0); i < iend; ++i)
          ofs_sv << "," << data(i);
        ofs_sv << "]\n";
        ofs_sv << indent2 << "},\n";
      }
    };

    /// ignition delay time
    if (is_idt_write) {
      ofs_idt << "{\n";
    }
    auto write_idt = [&ofs_idt, is_idt_write, indent](const real_type_1d_view_host &data) {
      if (is_idt_write) {
        ofs_idt << std::scientific << std::setprecision(15);
        ofs_idt << indent << "\"ignition delay time\" : {\n";
        ofs_idt << indent << indent << "\"data\" : [ ";
        ofs_idt << data(0);
        for (ordinal_type i = 1, iend = data.extent(0); i < iend; ++i)
          ofs_idt << "," << data(i);
        ofs_idt << "]\n";
        ofs_idt << indent << "}\n";
      }
    };

    /// sensitivity to ignition delay time
    if (is_s2idt_write) {
      /// TODO:: R,P are empty but need to add for consistency
      ofs_s2idt << "{\n" << indent << "\"variable name\" : [ \"temperature\"";
      const auto species_name = kmcd_host.speciesNames;
      for (ordinal_type i = 0, iend = species_name.extent(0); i < iend; ++i) {
        ofs_s2idt << ",\"" << std::string(&species_name(i, 0)) << "\"";
      }
      ofs_s2idt << "],\n" << indent << "\"parameter name\" : [ ";
      ofs_s2idt << "\"" << 0 << "\"";
      for (ordinal_type i = 1, iend = kmcd_host.nReac; i < iend; ++i) {
        ofs_s2idt << ",\"" << i << "\"";
      }
      ofs_s2idt << "],\n";
      ofs_s2idt << indent << "\"number of samples\" :" << n_samples << ",\n";
      ofs_s2idt << indent << "\"state vector\" : [\n";
    }
    auto write_s2idt = [&ofs_s2idt, is_s2idt_write, indent2, indent3](const ordinal_type s, const real_type t,
                                                                      const real_type dt,
                                                                      const real_type_2d_view_host &data) {
      if (is_s2idt_write) {
        ofs_s2idt << std::scientific << std::setprecision(15);
        ofs_s2idt << indent2 << "{\n";
        ofs_s2idt << indent3 << "\"sample\" : " << s << ",\n";
        ofs_s2idt << indent3 << "\"t\" : " << t << ",\n";
        ofs_s2idt << indent3 << "\"dt\" : " << dt << ",\n";
        ofs_s2idt << indent3 << "\"data\" : [ ";
        const ordinal_type m = data.extent(0), n = data.extent(1);
        ofs_s2idt << data(0, 0);
        for (ordinal_type ij = 1, ijend = m * n; ij < ijend; ++ij) {
          const ordinal_type i = ij / n, j = ij % n;
          ofs_s2idt << "," << data(i, j);
        }
        ofs_s2idt << "]\n";
        ofs_s2idt << indent2 << "},\n";
      }
    };

    // write generic data
    auto write_data = [indent2, indent3](std::ofstream &ofs, const ordinal_type s, const real_type_1d_view_host &data) {
      ofs << std::scientific << std::setprecision(15);
      ofs << indent2 << "{\n";
      ofs << indent3 << "\"sample\" : " << s << ",\n";
      ofs << indent3 << "\"data\" : [ ";
      ofs << data(0);
      for (ordinal_type i = 1, iend = data.extent(0); i < iend; ++i)
        ofs << "," << data(i);
      ofs << "]\n";
      ofs << indent2 << "},\n";
    };

    auto kmcds = TChem::createGasKineticModelConstData<device_type>(kmds);
    using team_policy_type = typename TChem::UseThisTeamPolicy<TChem::exec_space>::type;

    /// function evaluation
    ///
    /// evaluation of functions: source terms, jacobians, reaction constants ... mainly use for testing
    // reaction rate constants
    boost::property_tree::ptree node;
    try {
      if (validate_optional(root, "function evaluation", node)) {
        // list of function evaluations
        const auto exec_space_instance = exec_space();
        team_policy_type team_policy(exec_space_instance, n_samples, Kokkos::AUTO());
        if (team_size > 0 && vector_size > 0) {
          team_policy = team_policy_type(exec_space_instance, n_samples, team_size, vector_size);
        } // end team_size

        boost::property_tree::ptree gas_rate_constants_node;
        if (validate_optional(node, "gas reaction rate constants", gas_rate_constants_node)) {
          const ordinal_type level = 1;
          const ordinal_type per_team_extent = KForwardReverse::getWorkSpaceSize(kmcd_host);
          const ordinal_type per_team_scratch = TChem::Scratch<real_type_1d_view>::shmem_size(per_team_extent);

          real_type_3d_view kfor_krev("kfor_krev", 2, n_samples, kmcd_host.nReac);
          const auto kfor = Kokkos::subview(kfor_krev, 0, Kokkos::ALL(), Kokkos::ALL());
          const auto krev = Kokkos::subview(kfor_krev, 1, Kokkos::ALL(), Kokkos::ALL());

          team_policy.set_scratch_size(level, Kokkos::PerTeam(per_team_scratch));
          KForwardReverse::runDeviceBatch(team_policy, sv, kfor, krev, kmcds);

          boost::property_tree::ptree output_node;
          validate_required(gas_rate_constants_node, "output", output_node);
          std::string filename;
          validate_required(output_node, "file name", filename);
          std::string output_filename = replace_env_variable(filename);
          // if we do not want to save a file: use none for output
          if (filename != "none") {
            std::ofstream ofs_rrc;
            ofs_rrc.open(output_filename);
            {
              std::string msg("Error: fail to open file " + output_filename);
              TCHEM_CHECK_ERROR(!ofs_sv.is_open(), msg.c_str());
            }
            auto kfor_krev_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), kfor_krev);
            ofs_rrc << "{\n";
            ofs_rrc << indent << "\"number of samples\" :" << n_samples << ",\n";
            ofs_rrc << indent << "\"number of reactions\" :" << kmcd_host.nReac << ",\n";
            ofs_rrc << indent << "\"k-reverse-rate\" : [\n";

            for (ordinal_type i = 0, iend = n_samples; i < iend; ++i) {
              const real_type_1d_view_host krev_host_at_i = Kokkos::subview(kfor_krev_host, 1, i, Kokkos::ALL());
              write_data(ofs_rrc, i, krev_host_at_i);
            } // end samples
            ofs_rrc << indent << "{}],\n";

            ofs_rrc << indent << "\"k-forwad-rate\" : [\n";
            for (ordinal_type i = 0, iend = n_samples; i < iend; ++i) {
              const real_type_1d_view_host kfwd_host_at_i = Kokkos::subview(kfor_krev_host, 0, i, Kokkos::ALL());
              write_data(ofs_rrc, i, kfwd_host_at_i);
            } // end samples

            ofs_rrc << indent << "{}]\n}\n";
            ofs_rrc.close();
          }
        }
      }
    } catch (const std::exception &e) {
      std::cerr << "Error: exception is caught in function evaluation \n" << e.what() << "\n";
    }

    ///
    /// driver
    ///

    try {
      real_type_2d_view ign_delay_time("ignition_delay_time", 2, n_samples);
      auto ign_delay_time_host = Kokkos::create_mirror_view(ign_delay_time);

      Kokkos::deep_copy(ign_delay_time, real_type(-1));

      /// team policy
      const auto exec_space_instance = exec_space();
      team_policy_type team_policy(exec_space_instance, n_samples, Kokkos::AUTO());

      if (team_size > 0 && vector_size > 0) {
        team_policy = team_policy_type(exec_space_instance, n_samples, team_size, vector_size);
      }

      const ordinal_type level = 1;
      ordinal_type per_team_extent(0);

      if (reactor_type_name == "constant volume homogeneous batch reactor") {
        per_team_extent = TChem::ConstantVolumeIgnitionReactor::getWorkSpaceSize(solve_tla, kmcd_host);
      } else if (reactor_type_name == "constant pressure homogeneous batch reactor") {
        per_team_extent = TChem::IgnitionZeroD::getWorkSpaceSize(kmcd_host);
      } else {
        std::string msg("Error: not yet implemented: " + reactor_type_name);
        TCHEM_CHECK_ERROR(true, msg.c_str());
      }

      const ordinal_type per_team_scratch = TChem::Scratch<real_type_1d_view>::shmem_size(per_team_extent);
      team_policy.set_scratch_size(level, Kokkos::PerTeam(per_team_scratch));

      { /// time integration
        real_type_1d_view t("time", n_samples);
        Kokkos::deep_copy(t, tadv_default._tbeg);
        real_type_1d_view dt("delta time", n_samples);
        real_type_2d_view fac("fac", n_samples, n_equations);

        auto t_host = Kokkos::create_mirror_view(t);
        auto dt_host = Kokkos::create_mirror_view(dt);

        real_type tsum(0);
        Kokkos::RangePolicy<exec_space> range_policy(0, n_samples);

        ordinal_type iter(0);
        { /// initial state vector output
          Kokkos::deep_copy(t_host, t);
          Kokkos::deep_copy(dt_host, dt);
          Kokkos::deep_copy(sv_host, sv);
          for (ordinal_type i = 0, iend = n_samples; i < iend; ++i) {
            const real_type_1d_view_host sv_host_at_i = Kokkos::subview(sv_host, i, Kokkos::ALL());
            write_sv(i, t_host(i), dt_host(i), sv_host_at_i);

            const real_type_2d_view_host sz_host_at_i = Kokkos::subview(sz_host, i, Kokkos::ALL(), Kokkos::ALL());
            write_s2idt(i, t_host(i), dt_host(i), sz_host_at_i);
          }
        }

        for (; iter < max_num_kernel_launch && tsum <= tadv_default._tend * 0.9999; ++iter) {
          if (reactor_type_name == "constant volume homogeneous batch reactor") {
            ConstantVolumeIgnitionReactor::runDeviceBatch(team_policy, solve_tla, theta, tol_newton, tol_time, fac,
                                                          tadv, sv, sz, t, dt, sv, sz, kmcds);
          } else if (reactor_type_name == "constant pressure homogeneous batch reactor") {
            TChem::IgnitionZeroD::runDeviceBatch(team_policy, tol_newton, tol_time, fac, tadv, sv, t, dt, sv, kmcds);
          }

          /// test ignition
          Kokkos::parallel_for(
              range_policy, KOKKOS_LAMBDA(const ordinal_type &i) {
                if (ign_delay_time(0, i) < 0 && sv(i, 2) >= temperature_threshold_to_ignition_delay_time) {
                  ign_delay_time(0, i) = t(i);
                }
                /// TODO:: test with second derivatives
              });

          /// carry over time and dt computed in this step
          tsum = 0;
          Kokkos::parallel_reduce(
              range_policy,
              KOKKOS_LAMBDA(const ordinal_type &i, real_type &update) {
                tadv(i)._tbeg = t(i);
                tadv(i)._dt = dt(i);
                update += t(i);
              },
              tsum);
          Kokkos::fence();
          tsum /= real_type(n_samples);

          { /// state vector output
            Kokkos::deep_copy(t_host, t);
            Kokkos::deep_copy(dt_host, dt);
            Kokkos::deep_copy(sv_host, sv);
            Kokkos::deep_copy(sz_host, sz);
            if (verbose > 3) {
              std::cout << std::scientific << std::setprecision(6) << " sample 0, iter = " << std::setw(5) << iter
                        << " t = " << t_host(0) << " dt = " << dt_host(0) << " temperature = " << sv_host(0, 2) << "\n";
            }
            for (ordinal_type i = 0, iend = n_samples; i < iend; ++i) {
              const real_type_1d_view_host sv_host_at_i = Kokkos::subview(sv_host, i, Kokkos::ALL());
              write_sv(i, t_host(i), dt_host(i), sv_host_at_i);

              const real_type_2d_view_host sz_host_at_i = Kokkos::subview(sz_host, i, Kokkos::ALL(), Kokkos::ALL());
              write_s2idt(i, t_host(i), dt_host(i), sz_host_at_i);
            }
          }
        }

        {
          Kokkos::deep_copy(ign_delay_time_host, ign_delay_time);
          auto ign_delay_time_select_host = Kokkos::subview(ign_delay_time_host, 0, Kokkos::ALL());
          write_idt(ign_delay_time_select_host);
        }
      }
    } catch (const std::exception &e) {
      std::cerr << "Error: exception is caught running time integration\n" << e.what() << "\n";
    }

    ///
    /// closing
    ///
    if (is_sv_write) {
      ofs_sv << indent << "{}]\n}\n";
      ofs_sv.close();
    }
    if (is_idt_write) {
      ofs_idt << "}\n";
      ofs_idt.close();
    }
    if (is_s2idt_write) {
      ofs_s2idt << "{}]\n}\n";
      ofs_s2idt.close();
    }
  }
  Kokkos::finalize();

  return 0;
}
