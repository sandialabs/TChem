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
#include "TChem_Main_Util.hpp"
#include "boost/json/src.hpp"

int parse_json(const std::string &filename, boost::json::value &jin) {
  int r_val(0);
  try {
    std::ifstream file(filename);
    {
      std::string msg("Error: file is not open: " + filename);
      TCHEM_CHECK_ERROR(!file.is_open(), msg.c_str());
    }
    std::stringstream ss;
    ss << file.rdbuf();
    file.close();

    boost::json::error_code err;
    boost::json::parse_options option;
    option.allow_comments = true;
    option.allow_trailing_commas = true;

    jin = boost::json::parse(ss.str(), err, boost::json::storage_ptr(), option);
    {
      std::string msg("Error: boost json parsing error: " + err.message());
      TCHEM_CHECK_ERROR(err.value(), msg.c_str());
    }
  } catch (const std::exception &e) {
    std::cerr << "Error: exception is caught parsing JSON input\n" << e.what() << "\n";
    r_val = -1;
  }
  return r_val;
}

std::string replace_env_variable(const std::string &in) {
  std::string r_val(in);
  std::regex env("\\$\\{([^}]+)\\}");
  std::smatch match;
  while (std::regex_search(r_val, match, env)) {
    const char *s = getenv(match[1].str().c_str());
    const std::string var(s == NULL ? "" : s);
    /// workaround for the compiler gcc 8 on silver
    const int pos = match[0].first - r_val.begin();
    const int len = match[0].second - match[0].first;
    r_val.replace(pos, len, var.c_str(), var.length());

    // r_val.replace(match[0].first, match[0].second, var);
  }
  return r_val;
}

int validate(boost::property_tree::ptree &tree, boost::property_tree::ptree &root) {
  int r_val(0);
  try {
    const auto it = tree.find("tchem");
    TCHEM_CHECK_ERROR(it == tree.not_found(), "Error: tchem node does not exist");
    root = it->second;
    {
      boost::property_tree::ptree null_node;

      std::string n_samples;
      validate_optional(root, "number of samples", n_samples);

      boost::property_tree::ptree run_node;
      if (validate_optional(root, "run", run_node)) {
        std::string run_type, team_size, vector_size;
        validate_required(run_node, "team size", team_size);
        validate_required(run_node, "vector size", vector_size);
      }

      boost::property_tree::ptree reactor_node;
      validate_required(root, "reactor", reactor_node);
      {
        std::string reactor_type;
        validate_required(reactor_node, "type", reactor_type);
      }
      const auto reactor_type_name = reactor_node.get<std::string>("type");
      const bool is_gas_kinetic_model_required = (reactor_type_name == "constant volume homogeneous batch reactor" ||
                                                  reactor_type_name == "constant pressure homogeneous batch reactor" ||
                                                  reactor_type_name == "transient continuous stirred tank reactor" ||
                                                  reactor_type_name == "plug flow reactor");
      const bool is_surface_kinetic_model_required =
          (reactor_type_name == "transient continuous stirred tank reactor" ||
           reactor_type_name == "plug flow reactor");
      const bool is_supported_reactor = (is_gas_kinetic_model_required || is_surface_kinetic_model_required);

      if (!is_supported_reactor) {
        std::string msg("Error: reactor type is not supported: " + reactor_type_name);
        TCHEM_CHECK_ERROR(true, msg.c_str());
      }

      if (is_gas_kinetic_model_required) {
        boost::property_tree::ptree gas_kinetic_model_node;
        validate_required(root, "gas kinetic model", gas_kinetic_model_node);
        {
          std::string gkm_type, gkm_input_file, gkm_thermo_file;
          validate_required(gas_kinetic_model_node, "type", gkm_type);
          validate_required(gas_kinetic_model_node, "input file name", gkm_input_file);
          if (gkm_type == "chemkin")
            validate_required(gas_kinetic_model_node, "thermo file name", gkm_thermo_file);
        }
      }

      if (is_surface_kinetic_model_required) {
        boost::property_tree::ptree surface_kinetic_model_node;
        validate_required(root, "surface kinetic model", surface_kinetic_model_node);
        {
          std::string skm_type, skm_input_file, skm_thermo_file;
          validate_required(surface_kinetic_model_node, "type", skm_type);
          validate_required(surface_kinetic_model_node, "input file name", skm_input_file);
          if (skm_type == "chemkin")
            validate_required(surface_kinetic_model_node, "thermo file name", skm_thermo_file);
        }
      }

      if (is_supported_reactor) {
        boost::property_tree::ptree time_integrator_node;
        validate_required(root, "time integrator", time_integrator_node);
        {
          std::string time_type;
          validate_required(time_integrator_node, "type", time_type);

          boost::property_tree::ptree newton_node;
          validate_required(time_integrator_node, "newton solver", newton_node);
          {
            double atol_newton, rtol_newton;
            validate_required(newton_node, "absolute tolerance", atol_newton);
            validate_required(newton_node, "relative tolerance", rtol_newton);

            int num_jacobian_evals_per_interval_newton, max_num_iterations_newton;
            validate_required(newton_node, "jacobian evaluation interval", num_jacobian_evals_per_interval_newton);
            validate_required(newton_node, "max number of newton iterations", max_num_iterations_newton);
          }

          boost::property_tree::ptree time_node;
          validate_required(time_integrator_node, "time", time_node);
          {
            double tbeg, tend, dtmin, dtmax, atol_time, rtol_time;
            validate_required(time_node, "time begin", tbeg);
            validate_required(time_node, "time end", tend);
            validate_required(time_node, "min time step size", dtmin);
            validate_required(time_node, "max time step size", dtmax);
            validate_required(time_node, "absolute tolerance", atol_time);
            validate_required(time_node, "relative tolerance", rtol_time);

            int max_num_kernel_launch, num_internal_time_iterations, num_outer_time_iterations;
            validate_required(time_node, "max number of kernel launch", max_num_kernel_launch);
            validate_required(time_node, "number of internal time iterations", num_internal_time_iterations);
            validate_required(time_node, "number of outer time iterations", num_outer_time_iterations);
          }
        }
      }

      boost::property_tree::ptree sensitivity_node;
      if (validate_optional(root, "sensitivity to ignition delay time", sensitivity_node)) {
        std::string sensitivity_type;
        validate_required(sensitivity_node, "type", sensitivity_type);

        double theta;
        validate_required(sensitivity_node, "theta", theta);
      }

      boost::property_tree::ptree function_evaluation_node;
      if (validate_optional(root, "function evaluation", function_evaluation_node)) {
        boost::property_tree::ptree gas_reaction_rates_node;
        if (validate_optional(function_evaluation_node, "gas reaction rate constants", gas_reaction_rates_node)) {
          boost::property_tree::ptree gas_reaction_rates_output_node;
          validate_required(gas_reaction_rates_node, "output", gas_reaction_rates_output_node);
          std::string file_name;
          validate_required(gas_reaction_rates_output_node, "file name", file_name);
        }
      }

      boost::property_tree::ptree input_node;
      validate_required(root, "input", input_node);
      {
        boost::property_tree::ptree state_vector_node;
        validate_required(input_node, "state vector", state_vector_node);
        {
          std::string filename;
          validate_required(state_vector_node, "file name", filename);
        }
      }

      boost::property_tree::ptree output_node;
      if (validate_optional(root, "output", output_node)) {
        boost::property_tree::ptree state_vector_node;
        if (validate_optional(output_node, "state vector", state_vector_node)) {
          std::string filename;
          validate_required(state_vector_node, "file name", filename);
        }

        boost::property_tree::ptree ignition_delay_time_node;
        if (validate_optional(output_node, "ignition delay time", ignition_delay_time_node)) {
          double threshold_temperature;
          validate_required(ignition_delay_time_node, "threshold temperature", threshold_temperature);

          std::string filename;
          validate_required(ignition_delay_time_node, "file name", filename);
        }

        boost::property_tree::ptree sensitivity_output_node;
        if (validate_optional(output_node, "sensitivity to ignition delay time", sensitivity_output_node)) {
          std::string filename;
          validate_required(sensitivity_output_node, "file name", filename);
        }
      }
    }
  } catch (const std::exception &e) {
    r_val = -1;
    std::cerr << "Error: exception is caught validating input\n" << e.what() << "\n";
  }

  return r_val;
}
