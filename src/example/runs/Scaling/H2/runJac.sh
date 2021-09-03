
exec=$TCHEM_INSTALL_PATH/example/TChem_IgnitionZeroDJacobians.x
inputs=$TCHEM_INSTALL_PATH/example/data/H2/
chemfile=$inputs"chem.inp"
thermfile=$inputs"therm.dat"
use_sample_format=false
outputfile=".dat"
samplefile="inputGas.dat"
output_times=HostTimes/
mkdir $output_times
for NT in 20 40 80 160
do
for N in 1 10 50 100 500 1000 5000 10000 50000 100000 200000 300000 
do
 output_file_times=$output_times"wall_times_"$NT"_"$N".dat"
 run_this="OMP_NUM_THREADS=$NT OMP_PLACES=threads  OMP_PROC_BIND=spread $exec --batchsize=$N --outputfile=$outputfile --verbose=false --output-file-times=$output_file_times --use-sample-format=$use_sample_format --chemfile=$chemfile --thermfile=$thermfile --inputfile=$samplefile"
 echo $run_this
 eval $run_this
done
done
