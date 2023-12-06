#include <torch/extension.h>
#include <cmath>


template <typename scalar_t>
__global__ void qk_fw_kernel(
    const torch::PackedTensorAccessor<scalar_t, 4, torch::RestrictPtrTraits, size_t> queries,
    const torch::PackedTensorAccessor<scalar_t, 4, torch::RestrictPtrTraits, size_t> keys,
    torch::PackedTensorAccessor<scalar_t, 4, torch::RestrictPtrTraits, size_t> attn_weight,
    int height,
    int width,
    int kernel_size
){
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x < (queries.size(0)* queries.size(1))){
        const int y = blockIdx.y * blockDim.y + threadIdx.y;
        if (y < queries.size(2)){
            const int z = blockIdx.z * blockDim.z + threadIdx.z;
            if (z < kernel_size * kernel_size){
                const int b = x / queries.size(1);
                const int h = x - b * queries.size(1);
                const int ki = z / kernel_size;
                const int kj = z - ki * kernel_size;
                const int i = y / width;
                const int j = y - i * width;
                const int ni = i+ki-(kernel_size-1)/2;
                const int nj = j+kj-(kernel_size-1)/2;

                scalar_t updt = scalar_t(0);
                if (((ni>=0) && (ni<height))&& ((nj>=0) && (nj<width))){
                    const int key_y = ni*width+nj;
                    #pragma unroll
                    for (int dimOffset=0; dimOffset < queries.size(3); ++dimOffset)
                        updt += queries[b][h][y][dimOffset] * keys[b][h][key_y][dimOffset];
                }
                else{
                    updt = scalar_t(-INFINITY);
                }
                attn_weight[b][h][y][z]=updt;

            }

        }
    }
}


torch::Tensor qk_fw_cu(
    const torch::Tensor queries,
    const torch::Tensor keys,
    int height,
    int width,
    int kernel_size,
    int cuda_threads
){
    TORCH_CHECK((cuda_threads>0)&&(cuda_threads<=1024),"The value of CUDA_NUM_THREADS should between 1 and 1024");
    TORCH_CHECK(queries.size(0) == keys.size(0), "Query and Key should have same Batch_Size");
    TORCH_CHECK(queries.size(1) == keys.size(1), "Query and Key should have same Head Nums");
    TORCH_CHECK(queries.size(2) == keys.size(2), "Query and Key should have same Pixel Nums");
    TORCH_CHECK(queries.size(3) == keys.size(3), "Query and Key should have same Head Dims");
    const int B= queries.size(0), N = queries.size(1), L = queries.size(2), C = queries.size(3);

    const int attention_span = kernel_size* kernel_size;
    const int KERNELTHREADS = min(cuda_threads, attention_span);
    const int PIXELTHREADS = min(int(cuda_threads / KERNELTHREADS), L);
    const int BATCHTHREADS = max(1, cuda_threads / (PIXELTHREADS * KERNELTHREADS));
    
    torch::Tensor attn_weight = torch::empty({B, N, L, attention_span}, queries.options());

    const dim3 threads(BATCHTHREADS, PIXELTHREADS, KERNELTHREADS);
    const dim3 blocks(((B*N)+threads.x-1)/threads.x, (L+threads.y-1)/threads.y, (attention_span+threads.z-1)/threads.z);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(queries.type(), "qk_fw_cu", 
    ([&] {
        qk_fw_kernel<scalar_t><<<blocks, threads>>>(
            queries.packed_accessor<scalar_t, 4, torch::RestrictPtrTraits, size_t>(),
            keys.packed_accessor<scalar_t, 4, torch::RestrictPtrTraits, size_t>(),
            attn_weight.packed_accessor<scalar_t, 4, torch::RestrictPtrTraits, size_t>(),
            height,
            width,
            kernel_size
        );
    }));

    return attn_weight;
}


