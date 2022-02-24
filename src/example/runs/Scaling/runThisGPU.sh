one=1
ksize=$(eval echo "{0..$(expr ${#league[@]} - $one)}")
isize=$(eval echo "{0..$(expr ${#team[@]} - $one)}")

echo $ksize
echo $isize

for k in $ksize
do
    for i in $isize
    do
        N=${Nproblem[k]}
        league_size=${league[k]}
        team_size=${team[i]}
        vector_size=${vector[i]}

        echo "num problems " $N "league size" $league_size  "team_size" $team_size "vector_size" $vector_size 
        output=$output_times/"Times_nBacth"$N"_V"$vector_size"T"$team_size
        output_file_times=$output".dat"

        run_this="OMP_NUM_THREADS=1 OMP_PLACES=threads  OMP_PROC_BIND=spread $exec \
--batchsize=$N \
--league_size=$league_size \
--team_thread_size=$team_size \
--vector_thread_size=$vector_size \
--run_analytic_Jacobian=$run_analytic_Jacobian \
--run_numerical_Jacobian=$run_numerical_Jacobian \
--run_numerical_Jacobian_fwd=$run_numerical_Jacobian_fwd \
--run_sacado_Jacobian=$run_sacado_Jacobian \
--run_source_term=$run_source_term \
--outputfile=$outputfile \
--output-file-times=$output_file_times \
--use-sample-format=$use_sample_format \
--chemfile=$chemfile \
--thermfile=$thermfile \
--inputfile=$samplefile \
--use-shared-workspace=$use_shared_work_space \
--verbose=false"
        echo $run_this
        eval $run_this
    done
done
