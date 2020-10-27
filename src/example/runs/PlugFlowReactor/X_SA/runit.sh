
exec=$TCHEM_INSTALL_PATH/example/TChem_PlugFlowReactor.x
inputs=$TCHEM_INSTALL_PATH/example/data/plug-flow-reactor/X/
inputs_conditions=inputs/
use_prefixPath=false

chemfile=$inputs"chem.inp"
thermfile=$inputs"therm.dat"
chemSurffile=$inputs"chemSurf.inp"
thermSurffile=$inputs"thermSurf.dat"
samplefile=$inputs_conditions"sample.dat"
inputSurffile=$inputs_conditions"inputSurf.dat"
inputVelocityfile=$inputs_conditions"inputVelocity.dat"

save=1
dzmin=1e-12
dzmax=1e-5
zend=0.025
max_newton_iterations=100
max_z_iterations=1000
atol_newton=1e-12
rtol_newton=1e-6
tol_z=1e-5
transient_initial_condition=true
initial_condition=false
Area=0.00053
Pcat=0.025977239243415308

$exec --use_prefixPath=$use_prefixPath --chemfile=$chemfile --thermfile=$thermfile --chemSurffile=$chemSurffile --thermSurffile=$thermSurffile --samplefile=$samplefile --inputSurffile=$inputSurffile --inputVelocityfile=$inputVelocityfile --initial_condition=$initial_condition --transient_initial_condition=$transient_initial_condition --Area=$Area  --Pcat=$Pcat --tol-z=$tol_z --atol-newton=$atol_newton --rtol-newton=$rtol_newton --dzmin=$dzmin --max-newton-iterations=$max_newton_iterations --output_frequency=$save --dzmax=$dzmax --zend=$zend --max-z-iterations=$max_z_iterations

