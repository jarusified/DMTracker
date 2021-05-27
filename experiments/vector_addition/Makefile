CUCOMP = nvcc
CFLAGS =
CUDATOOLKIT_HOME=/home/suraj/Work/llnl/spack/opt/spack/linux-ubuntu20.04-skylake/gcc-9.3.0/cuda-11.1.0-5zknzvukfvdi2ihfhj25fcqj4gktqesr/
CUPTI_PATH=$(CUDATOOLKIT_HOME)/extras/CUPTI
INCLUDES=-I$(CUPTI_PATH)/include

ifeq ($(OS),Windows_NT)
    export PATH := $(PATH):$(CUPTI_PATH)/libWin32:$(CUPTI_PATH)/libx64
    LIBS= -lcuda -L $(CUPTI_PATH)/libWin32 -L $(CUPTI_PATH)/libx64 -lcupti
    OBJ = obj
else
    ifeq ($(OS), Darwin)
        export DYLD_LIBRARY_PATH := $(DYLD_LIBRARY_PATH):$(CUPTI_PATH)/lib
        LIBS= -Xlinker -framework -Xlinker cuda -L $(CUPTI_PATH)/lib -lcupti
    else
        export LD_LIBRARY_PATH := $(LD_LIBRARY_PATH):$(CUPTI_PATH)/lib:$(CUPTI_PATH)/lib64
        LIBS= -lcuda -L $(CUPTI_PATH)/lib -L $(CUPTI_PATH)/lib64 -lcupti
    endif
    OBJ = o
endif

# run: vector_addition.$(OBJ)
# 	$(CUCOMP) $(CFLAGS) vector_addition.$(OBJ) -o vector_addition

activity_trace_async: activity_trace_async.$(OBJ) vector_addition.$(OBJ)
	nvcc -o $@ $^ $(LIBS)

activity_trace_async.$(OBJ): activity_trace_async.cpp
	nvcc -c $(INCLUDES) $<

vector_addition.$(OBJ): vector_addition.cu
	$(CUCOMP) $(CFLAGS) -c $(INCLUDES) $<

run: activity_trace_async
	./$<

.PHONY: clean cleanall

clean:
	rm -f vector_addition *.o

cleanall:
	rm -f vector_addition *.o vec_add_cuda.* *.nvvp

profile_all: 
	nv-nsight-cu-cli --metrics "regex:.*" ./vector_addition >> ./all_metrics.txt

generate-qdrep:
	nsys launch ./vector_addition

convert-qdrep:
	nsys export -t hdf filename.qdrep