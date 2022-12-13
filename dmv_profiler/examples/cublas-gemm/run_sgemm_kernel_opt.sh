# SGEMM - Kernel optimization.
if [ ! -d "sgemm-kernel-opt" ]; then
  echo "sgemm-kernel-opt folder does not exist. Creating it!"
  mkdir sgemm-kernel-opt
fi

for size in {1..5}
do 
    for kernel in {0..5}
    do 
	    ./gemm-opt -s $size --kernel-version $kernel --traceFile sgemm-kernel-opt/kernel-opt-$size-$kernel-trace.json --metricsFile sgemm-kernel-opt/kernel-opt-$size-$kernel-metrics.json --fill-strategy "column-format"
    done
done
