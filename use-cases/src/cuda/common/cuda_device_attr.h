/* This header file is used for Tesla V100-SXM2-16GB. */
#ifndef __CUDA_DEVICE_ATTR_H__
#define __CUDA_DEVICE_ATTR_H__

#define CUDA_DEVICE_NUM    4
#define CUDA_DEVICE_ID    0
#define CUDA_DEVICE_NAME    "Tesla V100-SXM2-16GB"
#define CUDA_MAJOR_VERSION    10
#define CUDA_MINOR_VERSION    2
#define CUDA_MAJOR_CAPABILITY    7
#define CUDA_MINOR_CAPABILITY    0
#define GLOBAL_MEM    16911433728
#define SM_COUNT    80
#define CUDA_CORES_PER_SM    64
#define CUDA_CORES    5120
#define MEM_BUS_WIDTH    4096
#define L2_CACHE_SIZE    6291456
#define MAX_TEXTURE_1D_DIM    131072
#define MAX_TEXTURE_2D_X    131072
#define MAX_TEXTURE_2D_Y    65536
#define MAX_TEXTURE_3D_X    16384
#define MAX_TEXTURE_3D_Y    16384
#define MAX_TEXTURE_3D_Z    16384
#define MAX_LAYERED_1D_TEXTURE_SIZE    32768
#define MAX_LAYERED_1D_TEXTURE_LAYERS    2048
#define MAX_LAYERED_2D_TEXTURE_SIZE_X    32768
#define MAX_TEXTURE_2D_TEXTURE_SIZE_Y    32768
#define MAX_TEXTURE_2D_TEXTURE_LAYERS    2048
#define CONST_MEM    65536
#define SHARED_MEM_PER_BLOCK    49152
#define SHARED_MEM_PER_SM    98304
#define SHARED_MEMORY_BANKS    32
#define SHARED_MEMORY_BANK_BANDWIDTH    4 // Each bank has a bandwidth of 32 bits per clock cycle (no doc)
#define REGS_PER_BLOCK    65536
#define WARP_SIZE    32
#define MAX_THREADS_PER_SM    2048
#define MAX_THREADS_PER_BLOCK    1024
#define MAX_THREADS_DIM_X    1024
#define MAX_THREADS_DIM_Y    1024
#define MAX_THREADS_DIM_Z    64
#define MAX_GRIDS_DIM_X    2147483647
#define MAX_GRIDS_DIM_Y    65535
#define MAX_GRIDS_DIM_Z    65535
#define MEM_PITCH    1024
#define TEXTURE_ALIGNMENT    512

#define PAGE_LOCKED_MEM
#define UVA
#define MANAGED_MEM
#define COMPUTE_PREEMPTION
#define COOP_KERNEL
#define MULTI_DEVICE_COOP_KERNEL

#endif