exec=$TCHEM_INSTALL_PATH/example/TChem_IgnitionZeroDJacobians.x
inputs=$TCHEM_INSTALL_PATH/example/data/H2/
chemfile=$inputs"chem.inp"
thermfile=$inputs"therm.dat"
use_sample_format=false
outputfile=".dat"
samplefile="inputGas.dat"
Nproblem=(1 10 50 100 500 1000 5000 10000 50000 100000 200000 300000)
league=(1 10 50 100 500 1000 5000 10000 50000 100000 200000 300000)


