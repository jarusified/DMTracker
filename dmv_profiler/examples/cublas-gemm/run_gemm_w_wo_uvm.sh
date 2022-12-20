# Data Filling strategy.
if [ ! -d "gemm-w-wo-uvm" ]; then
  echo "gemm-w-wo-uvm folder does not exist. Creating it!"
  mkdir gemm-w-wo-uvm
else;
  rm gemm-w-wo-uvm
fi

lstopo --no-useless-caches --no-i-caches --whole-io -p --of svg > topology.svg

fill_strategy="default"

for size in {1..5}
do
  echo $size
  # Explicit memory transfer
  ./gemm-opt --fill-strategy $strategy --traceFile gemm-w-wo-uvm/explicit-$size.json  --metricsFile gemm-w-wo-uvm/explicit-$size.csv --size $size 

  # UVM data transfer modes.
  for strategy in "uvm" "uvm-prefetch" "uvm-advise" "uvm-prefetch-advise"
  do
    ./gemm-opt --fill-strategy $strategy --traceFile gemm-w-wo-uvm/$strategy-$size.json  --metricsFile gemm-w-wo-uvm/$strategy-$size.csv --size $size --$strategy
  done

done
