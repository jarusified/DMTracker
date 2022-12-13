# Data Filling strategy.
if [ ! -d "fill-strategy" ]; then
  echo "fill-strategy folder does not exist. Creating it!"
  mkdir fill-strategy
fi

for size in {1..5}
do 
    for strategy in "column-format" "default"
    do
	./gemm-opt --fill-strategy $strategy --traceFile fill-strategy/fill-$size-$strategy.json --size $size
    done
done

