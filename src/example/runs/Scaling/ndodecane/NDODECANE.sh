exec=$TCHEM_INSTALL_PATH/example/TChem_IgnitionZeroDJacobians.x
inputs=$TCHEM_INSTALL_PATH/example/data/ndodecane/
chemfile=$inputs"chem_ndodecane.inp"
thermfile=$inputs"therm_ndodecane.dat"
use_sample_format=false
outputfile=".dat"
samplefile="inputGas.dat"
output_times=HostTimes/
mkdir $output_times

Nproblem=(1 10 50 100 400 800 2000 4000 8000 16000)
league=(1 10 50 100 400 800 2000 4000 8000 16000)
