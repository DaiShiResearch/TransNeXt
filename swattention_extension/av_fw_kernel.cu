#include <torch/extension.h>
#include <cmath>


template <typename scalar_t>
__global__ void av_fw_kernel(
    const torch::PackedTensorAccessor<scalar_t, 4, torch::RestrictPtrTraits, size_t> attn_weight,
    const torch::PackedTensorAccessor<scalar_t, 4, torch::RestrictPtrTraits, size_t> values,
    torch::PackedTensorAccessor<scalar_t, 4, torch::RestrictPtrTraits, size_t> output,
    int height,
    int width,
    int kernel_size
){
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x < (values.size(0)* values.size(1))){
        const int y = blockIdx.y * blockDim.y + threadIdx.y;
        if (y < values.size(2)){
            const int z = blockIdx.z * blockDim.z + threadIdx.z;
            if (z < values.size(3)){
                const int b = x / values.size(1);
                const int h = x - b * values.size(1);
                const int i = y / width;
                const int j = y - i * width;
                const int start_i = i-(kernel_size-1)/2;
                const int start_j = j-(kernel_size-1)/2;
                scalar_t updt = scalar_t(0);
                int k_offset=0;

                #pragma unroll
                for (int current_i=start_i; current_i<(start_i+kernel_size); ++current_i){
                    #pragma unroll
                    for (int current_j=start_j; current_j<(start_j+kernel_size); ++current_j){
                        if (((current_i>=0) && (current_i<height))&& ((current_j>=0) && (current_j<width))){
                            const int current_offset=current_i*width+current_j;
                            updt += attn_weight[b][h][y][k_offset] * values[b][h][current_offset][z]; 
                        }
                        ++k_offset;
                    }
                }
                output[b][h][y][z]=updt; 

            }

        }
    }
}


torch::Tensor av_fw_cu(
    const torch::Tensor attn_weight,
    const torch::Tensor values,
    int height,
    int width,
    int kernel_size,
    int cuda_threads
){
    TORCH_CHECK((cuda_threads>0)&&(cuda_threads<=1024),"The value of CUDA_NUM_THREADS should between 1 and 1024");
    TORCH_CHECK(attn_weight.size(0) == values.size(0), "Attention Weights and Value should have same Batch_Size");
    TORCH_CHECK(attn_weight.size(1) == values.size(1), "Attention Weights and Value should have same Head Nums");
    TORCH_CHECK(attn_weight.size(2) == values.size(2), "Attention Weights and Value should have same Pixel Nums");

    const int B= values.size(0), N = values.size(1), L = values.size(2), C = values.size(3);

    const int attention_span = kernel_size* kernel_size;
    const int DIMTHREADS = min(cuda_threads, C);
    const int PIXELTHREADS = min(int(cuda_threads / DIMTHREADS), L);
    const int BATCHTHREADS = max(1, cuda_threads / (PIXELTHREADS * DIMTHREADS));
    
    torch::Tensor output = torch::empty({B, N, L, C}, attn_weight.options());

    const dim3 threads(BATCHTHREADS, PIXELTHREADS, DIMTHREADS);
    const dim3 blocks(((B*N)+threads.x-1)/threads.x, (L+threads.y-1)/threads.y, (C+threads.z-1)/threads.z);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(attn_weight.type(), "av_fw_cu", 
    ([&] {
        av_fw_kernel<scalar_t><<<blocks, threads>>>(
            attn_weight.packed_accessor<scalar_t, 4, torch::RestrictPtrTraits, size_t>(),
            values.packed_accessor<scalar_t, 4, torch::RestrictPtrTraits, size_t>(),
            output.packed_accessor<scalar_t, 4, torch::RestrictPtrTraits, size_t>(),
            height,
            width,
            kernel_size
        );
    }));

    return output;
}

