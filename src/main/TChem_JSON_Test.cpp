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

#include "TChem.hpp"
#include "TChem_CommandLineParser.hpp"
#include "TChem_Main_Util.hpp"

int main(int argc, char *argv[]) {

  std::string input_filename("input.json");
  std::string output_filename("ouput.json");
  int verbose(1);
  TChem::CommandLineParser opts("tchem-json-test.x");
  opts.set_option<std::string>("input", "input json filename", &input_filename);
  opts.set_option<std::string>("ouput", "sanitized output json filename ", &output_filename);
  opts.set_option<int>("verbose", "verbose output level", &verbose);

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
  boost::json::value jin;
  boost::property_tree::ptree tree;
  try {
    std::string filename = replace_env_variable(input_filename);
    const int r_val = parse_json(filename, jin);
    TCHEM_CHECK_ERROR(r_val, "Error: fails to parse JSON");
  } catch (const std::exception &e) {
    std::cerr << "Error: exception is caught parsing json\n"
              << e.what() << ", " << __FILE__ << ", " << __LINE__ << "\n";
  }

  try {
    std::stringstream ss;
    ss << jin;
    boost::property_tree::json_parser::read_json(ss, tree);
  } catch (const std::exception &e) {
    std::cerr << "Error: exception is caught converting property tree\n"
              << e.what() << ", " << __FILE__ << ", " << __LINE__ << "\n";
  }

  if (verbose) {
    std::cout << "-- Input JSON file after sanitizing\n";
    boost::property_tree::json_parser::write_json(std::cout, tree);
  }

  try {
    std::string filename = replace_env_variable(output_filename);
    std::ofstream ofs;
    ofs.open(filename);
    {
      std::string msg("Error: fail to open file " + filename);
      TCHEM_CHECK_ERROR(!ofs.is_open(), msg.c_str());
    }
    boost::property_tree::json_parser::write_json(ofs, tree);
  } catch (const std::exception &e) {
    std::cerr << "Error: exception is caught while creating sanitized output json\n"
              << e.what() << ", " << __FILE__ << ", " << __LINE__ << "\n";
  }

  return 0;
}
