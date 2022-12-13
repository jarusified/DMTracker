# Data Filling strategy.
if [ ! -d "gemm-w-wo-uvm" ]; then
  echo "gemm-w-wo-uvm folder does not exist. Creating it!"
  mkdir gemm-w-wo-uvm
fi

strategy="default"
size=1
./gemm-opt --fill-strategy $strategy --traceFile gemm-w-wo-uvm/explicit-64k.json --size $size

./gemm-opt --fill-strategy $strategy --traceFile gemm-w-wo-uvm/uvm-64k.json --size $size --uvm
