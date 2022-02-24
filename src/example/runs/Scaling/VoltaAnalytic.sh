output_times=VoltaJacAnalytic
mkdir $output_times

vector=(32 32 32 32)
team=(1 2 4 8)

run_sacado_Jacobian=false
run_source_term=false
run_analytic_Jacobian=true
run_numerical_Jacobian=false
run_numerical_Jacobian_fwd=false


