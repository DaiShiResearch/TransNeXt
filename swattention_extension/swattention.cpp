#include <torch/extension.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


torch::Tensor qk_fw_cu(
    const torch::Tensor queries,
    const torch::Tensor keys,
    int height,
    int width,
    int kernel_size,
    int cuda_threads
);

torch::Tensor qk_forward(
    const torch::Tensor queries,
    const torch::Tensor keys,
    int height,
    int width,
    int kernel_size,
    int cuda_threads
){
    CHECK_INPUT(queries);
    CHECK_INPUT(keys);

    return qk_fw_cu(queries, keys, height, width, kernel_size, cuda_threads);
}


std::vector<torch::Tensor> qk_bw_cu(
    const torch::Tensor d_attn_weight,
    const torch::Tensor queries,
    const torch::Tensor keys,
    int height,
    int width,
    int kernel_size,
    int cuda_threads
);

std::vector<torch::Tensor> qk_backward(
    const torch::Tensor d_attn_weight,
    const torch::Tensor queries,
    const torch::Tensor keys,
    int height,
    int width,
    int kernel_size,
    int cuda_threads
){
    CHECK_INPUT(d_attn_weight);
    CHECK_INPUT(queries);
    CHECK_INPUT(keys);

    return qk_bw_cu(d_attn_weight, queries, keys, height, width, kernel_size, cuda_threads);
}


std::vector<torch::Tensor> qk_rpb_bw_cu(
    const torch::Tensor d_attn_weight,
    const torch::Tensor queries,
    const torch::Tensor keys,
    int height,
    int width,
    int kernel_size,
    int cuda_threads
);

std::vector<torch::Tensor> qk_rpb_backward(
    const torch::Tensor d_attn_weight,
    const torch::Tensor queries,
    const torch::Tensor keys,
    int height,
    int width,
    int kernel_size,
    int cuda_threads
){
    CHECK_INPUT(d_attn_weight);
    CHECK_INPUT(queries);
    CHECK_INPUT(keys);

    return qk_rpb_bw_cu(d_attn_weight, queries, keys, height, width, kernel_size, cuda_threads);
}


torch::Tensor qk_rpb_fw_cu(
    const torch::Tensor queries,
    const torch::Tensor keys,
    const torch::Tensor rpb,
    int height,
    int width,
    int kernel_size,
    int cuda_threads
);

torch::Tensor qk_rpb_forward(
    const torch::Tensor queries,
    const torch::Tensor keys,
    const torch::Tensor rpb,
    int height,
    int width,
    int kernel_size,
    int cuda_threads
){
    CHECK_INPUT(queries);
    CHECK_INPUT(keys);
    CHECK_INPUT(rpb);

    return qk_rpb_fw_cu(queries, keys, rpb, height, width, kernel_size, cuda_threads);
}

torch::Tensor av_fw_cu(
    const torch::Tensor attn_weight,
    const torch::Tensor values,
    int height,
    int width,
    int kernel_size,
    int cuda_threads
);

torch::Tensor av_forward(
    const torch::Tensor attn_weight,
    const torch::Tensor values,
    int height,
    int width,
    int kernel_size,
    int cuda_threads
){
    CHECK_INPUT(attn_weight);
    CHECK_INPUT(values);

    return av_fw_cu(attn_weight, values, height, width, kernel_size, cuda_threads);
}


std::vector<torch::Tensor> av_bw_cu(
    const torch::Tensor d_output,
    const torch::Tensor attn_weight,
    const torch::Tensor values,
    int height,
    int width,
    int kernel_size,
    int cuda_threads
);

std::vector<torch::Tensor> av_backward(
    const torch::Tensor d_output,
    const torch::Tensor attn_weight,
    const torch::Tensor values,
    int height,
    int width,
    int kernel_size,
    int cuda_threads
){
    CHECK_INPUT(d_output);
    CHECK_INPUT(attn_weight);
    CHECK_INPUT(values);

    return av_bw_cu(d_output, attn_weight, values, height, width, kernel_size, cuda_threads);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    m.def("qk_forward", &qk_forward);
    m.def("qk_backward", &qk_backward);
    m.def("qk_rpb_forward", &qk_rpb_forward);
    m.def("qk_rpb_backward", &qk_rpb_backward);
    m.def("av_forward", &av_forward);
    m.def("av_backward", &av_backward);
}
