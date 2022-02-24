TCHEM_INSTALL_PATH=${HOME}/csp_clang_bld_develop/TChem-sacado/install
exec=$TCHEM_INSTALL_PATH/example/TChem_IgnitionZeroD.x
inputs=$TCHEM_INSTALL_PATH/example/data/ignition-zero-d/CO/
chemfile=$inputs"chem.inp"
thermfile=$inputs"therm.dat"

samplefile="sample.dat"
save=1
dtmin=1e-20
dtmax=1e-3
tend=10.0
max_time_iterations=1000
max_newton_iterations=20
atol_newton=1e-18
rtol_newton=1e-8
tol_time=1e-6
ignition_delay_file="IngitionDelayTime1.dat"
ignition_delay_file_Temp="IngitionDelayTime2.dat"
use_prefixPath=false
time_iterations_per_interval=10
OnlyComputeIgnDelayTime=false
verbose=false

OMP_NUM_THREADS=$NUM_THREADS OMP_PLACES=threads  OMP_PROC_BIND=spread $exec --verbose=$verbose --OnlyComputeIgnDelayTime=$OnlyComputeIgnDelayTime  --ignition-delay-time-w-threshold-temperature-file=$ignition_delay_file_Temp --ignition-delay-time-file=$ignition_delay_file --inputFileParamModifiers=$inputFileParamModifiers --chemfile=$chemfile --thermfile=$thermfile --samplefile=$samplefile --time-iterations-per-interval=$time_iterations_per_interval --use_prefixPath=$use_prefixPath --tol-time=$tol_time --atol-newton=$atol_newton --rtol-newton=$rtol_newton --dtmin=$dtmin --max-newton-iterations=$max_newton_iterations --output_frequency=$save --dtmax=$dtmax --tend=$tend --max-time-iterations=$max_time_iterations
