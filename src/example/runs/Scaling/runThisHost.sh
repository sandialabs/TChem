one=1
ksize=$(eval echo "{0..$(expr ${#Nproblem[@]} - $one)}")
isize=$(eval echo "{0..$(expr ${#Nthread[@]} - $one)}")

for k in $ksize
do
    N=${Nproblem[k]}
    for i in $isize
    do
        thread_size=${Nthread[i]}
        output_file_times=$output_times"wall_times_"$thread_size"_"$N".dat"
        run_this="OMP_NUM_THREADS=$thread_size OMP_PLACES=threads  OMP_PROC_BIND=spread $exec \
--batchsize=$N \
--outputfile=$outputfile \
--verbose=false \
--output-file-times=$output_file_times \
--use-sample-format=$use_sample_format \
--chemfile=$chemfile \
--thermfile=$thermfile \
--inputfile=$samplefile \
--run_numerical_Jacobian=false"
        echo $run_this
        eval $run_this
    done
done
