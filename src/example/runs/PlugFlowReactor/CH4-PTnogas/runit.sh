
exec=$TCHEM_INSTALL_PATH/example/TChem_PlugFlowReactor.x
inputs=$TCHEM_INSTALL_PATH/example/data/plug-flow-reactor/CH4-PTnogas/
save=1
dzmin=1e-12
dzmax=1e-5
zend=0.025
max_newton_iterations=20
max_z_iterations=310
atol_newton=1e-12
rtol_newton=1e-8
tol_z=1e-8
Area=0.00053
Pcat=0.025977239243415308
transient_initial_condition=true
initial_condition=true

$exec --prefixPath=$inputs --initial_condition=$initial_condition --transient_initial_condition=$transient_initial_condition --Area=$Area  --Pcat=$Pcat --tol-z=$tol_z --atol-newton=$atol_newton --rtol-newton=$rtol_newton --dzmin=$dzmin --max-newton-iterations=$max_newton_iterations --output_frequency=$save --dzmax=$dzmax --zend=$zend --max-z-iterations=$max_z_iterations

