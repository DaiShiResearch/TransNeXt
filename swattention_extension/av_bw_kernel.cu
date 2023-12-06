#include <torch/extension.h>
#include <cmath>

template <typename scalar_t>
__global__ void av_bw_kernel(
    const torch::PackedTensorAccessor<scalar_t, 4, torch::RestrictPtrTraits, size_t> d_output,
    const torch::PackedTensorAccessor<scalar_t, 4, torch::RestrictPtrTraits, size_t> values,
    torch::PackedTensorAccessor<scalar_t, 4, torch::RestrictPtrTraits, size_t> d_attn_weight,
    int height,
    int width,
    int kernel_size
){
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x < (d_output.size(0)* d_output.size(1))){
        const int y = blockIdx.y * blockDim.y + threadIdx.y;
        if (y < d_output.size(2)){
            const int z = blockIdx.z * blockDim.z + threadIdx.z;
            if (z < kernel_size * kernel_size){
                const int b = x / d_output.size(1);
                const int h = x - b * d_output.size(1);
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
                    for (int dimOffset=0; dimOffset < d_output.size(3); ++dimOffset)
                        updt += d_output[b][h][y][dimOffset] * values[b][h][key_y][dimOffset];
                }
                d_attn_weight[b][h][y][z]=updt;
            }

        }
    }
}

template <typename scalar_t>
__global__ void av_inverse_bw_kernel(
    const torch::PackedTensorAccessor<scalar_t, 4, torch::RestrictPtrTraits, size_t> attn_weight,
    const torch::PackedTensorAccessor<scalar_t, 4, torch::RestrictPtrTraits, size_t> d_output,
    torch::PackedTensorAccessor<scalar_t, 4, torch::RestrictPtrTraits, size_t> d_values,
    int height,
    int width,
    int kernel_size
){
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x < (d_values.size(0)* d_values.size(1))){
        const int y = blockIdx.y * blockDim.y + threadIdx.y;
        if (y < d_values.size(2)){
            const int z = blockIdx.z * blockDim.z + threadIdx.z;
            if (z < d_values.size(3)){
                const int b = x / d_values.size(1);
                const int h = x - b * d_values.size(1);
                const int i = y / width;
                const int j = y - i * width;
                const int q_start_i = i-kernel_size/2;
                const int q_end_i = i+1+(kernel_size-1)/2;
                const int q_start_j = j-kernel_size/2;
                const int q_end_j = j+1+(kernel_size-1)/2;
                scalar_t updt = scalar_t(0);
                int k_offset=kernel_size*kernel_size;
                #pragma unroll
                for (int current_i=q_start_i; current_i<q_end_i; ++current_i){
                    #pragma unroll
                    for (int current_j=q_start_j; current_j<q_end_j; ++current_j){
                        --k_offset;
                        if (((current_i>=0) && (current_i<height))&& ((current_j>=0) && (current_j<width))){
                            const int current_offset=current_i*width+current_j;
                            updt += attn_weight[b][h][current_offset][k_offset] * d_output[b][h][current_offset][z]; 
                        }            
                    }
                }
                d_values[b][h][y][z]=updt; 

            }

        }
    }
}

std::vector<torch::Tensor> av_bw_cu(
    const torch::Tensor d_output,
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

    const int A_KERNELTHREADS = min(cuda_threads, attention_span);
    const int A_PIXELTHREADS = min(int(cuda_threads / A_KERNELTHREADS), L);
    const int A_BATCHTHREADS = max(1, cuda_threads / (A_PIXELTHREADS * A_KERNELTHREADS));
    const dim3 A_threads(A_BATCHTHREADS, A_PIXELTHREADS, A_KERNELTHREADS);
    const dim3 A_blocks(((B*N)+A_threads.x-1)/A_threads.x, (L+A_threads.y-1)/A_threads.y, (attention_span+A_threads.z-1)/A_threads.z);

    const int V_DIMTHREADS = min(cuda_threads, C);
    const int V_PIXELTHREADS = min(int(cuda_threads / V_DIMTHREADS), L);
    const int V_BATCHTHREADS = max(1, cuda_threads / (V_PIXELTHREADS * V_DIMTHREADS));
    const dim3 V_threads(V_BATCHTHREADS, V_PIXELTHREADS, V_DIMTHREADS);
    const dim3 V_blocks(((B*N)+V_threads.x-1)/V_threads.x, (L+V_threads.y-1)/V_threads.y, (C+V_threads.z-1)/V_threads.z);
    
    torch::Tensor d_attn_weight = torch::empty({B, N, L, attention_span}, attn_weight.options());
    torch::Tensor d_values = torch::empty({B, N, L, C}, values.options());


    AT_DISPATCH_FLOATING_TYPES_AND_HALF(attn_weight.type(), "av_bw_cu", 
    ([&] {
        av_bw_kernel<scalar_t><<<A_blocks, A_threads>>>(
            d_output.packed_accessor<scalar_t, 4, torch::RestrictPtrTraits, size_t>(),
            values.packed_accessor<scalar_t, 4, torch::RestrictPtrTraits, size_t>(),
            d_attn_weight.packed_accessor<scalar_t, 4, torch::RestrictPtrTraits, size_t>(),
            height,
            width,
            kernel_size
        );
        av_inverse_bw_kernel<scalar_t><<<V_blocks, V_threads>>>(
            attn_weight.packed_accessor<scalar_t, 4, torch::RestrictPtrTraits, size_t>(),
            d_output.packed_accessor<scalar_t, 4, torch::RestrictPtrTraits, size_t>(),
            d_values.packed_accessor<scalar_t, 4, torch::RestrictPtrTraits, size_t>(),        
            height,
            width,
            kernel_size
        );
    }));

    return {d_attn_weight,d_values};
}