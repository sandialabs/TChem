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
#ifndef __TCHEM_MAIN_UTIL__
#define __TCHEM_MAIN_UTIL__

#include "TChem_Util.hpp"

#include "boost/json.hpp"
#include "boost/property_tree/json_parser.hpp"
#include "boost/property_tree/ptree.hpp"
#include "boost/optional.hpp"

int parse_json(const std::string &filename, boost::json::value &jin);
std::string replace_env_variable(const std::string &in);
int validate(boost::property_tree::ptree &tree, boost::property_tree::ptree &root);

template <typename T>
inline bool validate_optional(const boost::property_tree::ptree &node, const std::string key, T &val) {
  auto opt_val = node.get_optional<T>(key);
  bool r_val(false);
  if (opt_val) {
    val = opt_val.get();
    r_val = true;
  }
  return r_val;
}

template <typename T>
inline void validate_required(const boost::property_tree::ptree &node, const std::string key, T &val) {
  auto opt_val = node.get_optional<T>(key);
  if (opt_val)
    val = opt_val.get();
  else {
    std::string msg("Error: key does not exist: " + key);
    TCHEM_CHECK_ERROR(true, msg.c_str());
  }
}

template <>
inline bool validate_optional(const boost::property_tree::ptree &node, const std::string key,
                              boost::property_tree::ptree &val) {
  auto opt_val = node.get_child_optional(key);
  bool r_val(false);
  if (opt_val) {
    val = opt_val.get();
    r_val = true;
  }
  return r_val;
}

template <>
inline void validate_required(const boost::property_tree::ptree &node, const std::string key,
                              boost::property_tree::ptree &val) {
  auto opt_val = node.get_child_optional(key);
  if (opt_val)
    val = opt_val.get();
  else {
    std::string msg("Error: key does not exist: " + key);
    TCHEM_CHECK_ERROR(true, msg.c_str());
  }
}

#endif
