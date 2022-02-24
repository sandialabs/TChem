exec=$TCHEM_INSTALL_PATH/example/TChem_IgnitionZeroDJacobians.x
inputs=$TCHEM_INSTALL_PATH/example/data/n-butanol/
chemfile=$inputs"chem_nbutanol.inp"
thermfile=$inputs"therm_nbutanol.dat"
use_sample_format=false
outputfile=".dat"
samplefile="inputGas.dat"

Nproblem=(1 10 50 100 400 800 2000)
league=(1 10 50 100 400 800 2000)
