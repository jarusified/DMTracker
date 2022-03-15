exp_name=$1
memAlloc=("exp" "uvm" "uvm-prefetch" "uvm-advise" "uvm-prefetch-advise")

for s in [1, 2, 3, 4]; do 
    for exp in "${mem_alloc[@]}"
    do
        cmd="../use-cases/bin/cublas-gemm/cublas-gemm -s $s --$exp"
        python3 main.py --app_name $exp_name,s=$s,$exp --data_dir ./data/gemm --cmd $cmd
    done
done