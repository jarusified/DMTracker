CUCOMP = nvcc
CFLAGS =

run: vector_addition.o
	$(CUCOMP) $(CFLAGS) vector_addition.o -o vector_addition

vector_addition.o: vector_addition.cu
	$(CUCOMP) $(CFLAGS) -c vector_addition.cu

.PHONY: clean cleanall

clean:
	rm -f vector_addition *.o

cleanall:
	rm -f vector_addition *.o vec_add_cuda.* *.nvvp

profile_all: 
	nv-nsight-cu-cli --metrics "regex:.*" ./vector_addition >> ./all_metrics.txt