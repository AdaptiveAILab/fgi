# run with:
# nohup ./run_smnist.sh > run_smnist.log 2>&1 &

for in_size in 28 16 8 4 2 1; do
    for opt in none compile torchscript; do
        for grad in back fgi; do
            for run in 1 2 3 4 5 6 7 8 9 10; do
                python smnist.py --insize=${in_size} --opt=${opt} --grad=${grad} --run=${run}
            done  
        done  
    done
done

