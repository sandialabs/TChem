
# Checking PLOG reactions.

PLOG reactions are defined in a table of Arrhenius parameters at a set number of pressures. Reaction rates are evaluated at new pressures by linearly interpolating the log of the reaction rates. Because of this formulation, a negative rate constant could result from the linear interpolation. When this is the case, TChem throws an error and ends the reactor simulation with one of the following messages.  
```
Error: log(reaction rate) is nan. Sum of PLOG expressions results in a negative value (low range).
```
or
```
Error: log(reaction rate) is nan. Sum of PLOG expressions results in a negative value (high range).
```
When a mechanism has an error resulting in a negative rate at a feasible temperature and pressure, the user will need to fix the particular reaction that is causing this issue. To help the user, TChem has a tool, TCHEM_INSTALL_PATH/example/TChem_CheckPLOGreactions.x, that checks the rate constant across a range of temperature and pressures. This tool will throw the previous presented error message, if a negative rate constant is computed. This tool is faster to run than the reactor model and will test a range of temperatures/pressures.   

```
./TChem_CheckPLOGreactions.x --help
Usage: ./TChem_CheckPLOGreactions.x [options]
  options:
  --chemfile                    string    Chem file name e.g., chem.inp
                                          (default: --chemfile=data/propane/chem.inp)
  --echo-command-line           bool      Echo the command-line but continue as normal
  --factor-activation-energy    double    factor activation energy
                                          (default: --factor-activation-energy=1.0e+00)
  --factor-pre-exponential      double    factor pre-exponential
                                          (default: --factor-pre-exponential=1.0e+00)
  --factor-temperature-coeff    double    factor temperaturecoeff
                                          (default: --factor-temperature-coeff=1.0e+00)
  --final-pressure              double    final pressure, units Pascal
                                          (default: --final-pressure=4.0e+06)
  --final-temperature           double    final temperarure, units Kelvin
                                          (default: --final-temperature=4.0e+02)
  --help                        bool      Print this help message
  --initial-pressure            double    initial pressure, units Pascal
                                          (default: --initial-pressure=1.0e+02)
  --initial-temperature         double    initial temperarure, units Kelvin
                                          (default: --initial-temperature=3.00e+02)
  --inputsPath                  string    path to input files e.g., data/inputs
                                          (default: --inputsPath=data/propane/)
  --npoints-pressure            int       number of points sampled linearly between the initial and final pressure
                                          (default: --npoints-pressure=10)
  --npoints-temperature         int       number of points sampled linearly between the initial and final temperature
                                          (default: --npoints-temperature=10)
  --thermfile                   string    Therm file name e.g., therm.dat
                                          (default: --thermfile=data/propane/therm.dat)
  --verbose                     bool      If true, prints additional information
                                          (default: --verbose=true)
Description:
  Check PLOG reactions across a range of pressures and temperatures.
```
