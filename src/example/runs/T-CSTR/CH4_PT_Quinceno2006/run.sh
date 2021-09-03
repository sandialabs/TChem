
exec=$TCHEM_INSTALL_PATH/example/TChem_TransientContStirredTankReactor.x
inputs=inputs/

chemfile=$inputs"chemgri30.inp"
thermfile=$inputs"thermgri30.dat"
chemSurffile=$inputs"chemSurf.inp"
thermSurffile=$inputs"thermSurf.dat"
samplefile=$inputs"sample_phi1.dat"
inputSurffile=$inputs"inputSurf.dat"
save=1
dtmin=1e-20
dtmax=1e-3
tend=3
max_newton_iterations=20
max_time_iterations=10000
atol_newton=1e-18
rtol_newton=1e-8
tol_time=1e-8
atol_time=1e-12
Acat=1.347e-2
Vol=1.347e-1
mdotIn=1e-2
use_prefixPath=false
transient_initial_condition=true
initial_condition=false
time_iterations_per_interval=10
number_of_algebraic_constraints=0
outputfile="CSTRSolutionODE.dat"

$exec  --outputfile=$outputfile --number-of-algebraic-constraints=$number_of_algebraic_constraints --time-iterations-per-interval=$time_iterations_per_interval --atol-time=$atol_time --initial_condition=$initial_condition --transient_initial_condition=$transient_initial_condition --use_prefixPath=$use_prefixPath --chemfile=$chemfile --thermfile=$thermfile --chemSurffile=$chemSurffile --thermSurffile=$thermSurffile --samplefile=$samplefile --inputSurffile=$inputSurffile --mdotIn=$mdotIn --Acat=$Acat --Vol=$Vol --tol-time=$tol_time --atol-newton=$atol_newton --rtol-newton=$rtol_newton --dtmin=$dtmin --max-newton-iterations=$max_newton_iterations --output_frequency=$save --dtmax=$dtmax --tend=$tend --max-time-iterations=$max_time_iterations

