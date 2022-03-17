exp_name=$1
sizes=(1 2 3 4)
memAlloc=("exp" "uvm" "uvm-prefetch" "uvm-advise" "uvm-prefetch-advise")

for s in ${sizes[@]}; do 
    echo "Running problem size: $s"
    for exp in ${memAlloc[@]}; do
        echo "Memory allocation: $exp"
        cmd="../../../bin/cublas-gemm/cublas-gemm -s $s --$exp"
        if [ $exp = "exp" ]; then 
            cmd="../../../bin/cublas-gemm/cublas-gemm -s $s"
        fi 
        python3 ../../../../src/main.py --app_name $exp_name,s=$s,$exp --data_dir ../../data/gemm-new --cmd "$cmd"
    done
done