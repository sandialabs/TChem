
exec=$TCHEM_INSTALL_PATH/example/TChem_IgnitionZeroDJacobians.x
inputs=$TCHEM_INSTALL_PATH/example/data/ndodecane/
chemfile=$inputs"chem_ndodecane.inp"
thermfile=$inputs"therm_ndodecane.dat"
use_sample_format=false
outputfile=".dat"
samplefile="inputGas.dat"
output_times=DeviceJacAnalytic
run_sacado_Jacobian=false
run_source_term=false
run_analytic_Jacobian=true
run_numerical_Jacobian=false
run_numerical_Jacobian_fwd=false
mkdir $output_times


vector=(32 32 32 32)
team=(1 2 4 8)

for N in 1 10 50 100 500 1000 5000 10000 50000 100000 200000 300000 
do
 league_size=$N
 for i in $(eval echo "{0..3}")
 do
   vector_thread_size=${vector[i]}
   team_thread_size=${team[i]}
   echo "league size" $league_size  "vector_thread_size" $vector_thread_size "team_thread_size" $team_thread_size
   output=$output_times/"Times_nBacth"$N"_V"$vector_thread_size"T"$team_thread_size
   output_file_times=$output".dat"
   run_this="OMP_NUM_THREADS=1 OMP_PLACES=threads  OMP_PROC_BIND=spread $exec --batchsize=$N --league_size=$league_size --run_numerical_Jacobian=$run_numerical_Jacobian --run_numerical_Jacobian_fwd=$run_numerical_Jacobian_fwd --run_sacado_Jacobian=$run_sacado_Jacobian --run_source_term=$run_source_term --run_analytic_Jacobian=$run_analytic_Jacobian --vector_thread_size=$vector_thread_size --team_thread_size=$team_thread_size --outputfile=$outputfile --verbose=false --output-file-times=$output_file_times --use-sample-format=$use_sample_format --chemfile=$chemfile --thermfile=$thermfile --inputfile=$samplefile"
   echo "$run_this"
   eval $run_this
done
done
