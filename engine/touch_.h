#pragma once
#include <cstdint>

void touch1_dispatch(void*, const void*, uint64_t,
                     uint64_t, uint64_t, uint64_t,
                     uint64_t, cudaStream_t,int,int);

void touch2_dispatch(void*, const void*, uint64_t,
                     uint64_t, uint64_t, uint64_t,
                     uint64_t,uint64_t,uint64_t,
                     uint64_t,cudaStream_t,int,int);

void touch3_dispatch(void*, const void*, uint64_t,
                     uint64_t, uint64_t, uint64_t,
                     uint64_t,uint64_t,uint64_t,
                     uint64_t,uint64_t,uint64_t,
                     uint64_t,uint64_t,uint64_t,
                     cudaStream_t,int,int);
void touch4_dispatch(void*, const void*, uint64_t,
                     uint64_t, uint64_t, uint64_t,
                     uint64_t,uint64_t,uint64_t,
                     uint64_t,uint64_t,uint64_t,
                     uint64_t,uint64_t,uint64_t,
                     uint64_t,uint64_t,uint64_t,
                     uint64_t,uint64_t,
                     cudaStream_t,int,int);

