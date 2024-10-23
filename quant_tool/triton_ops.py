import torch
import triton
import triton.language as tl

def triton_kernel(input, qweight, scale):

    M, K = input.shape
    G,  N = scale.shape
    block_size = K // G

    output = torch.empty((M, N), device=input.device, dtype=input.dtype)

    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),)

    with torch.cuda.device(input.device):

        _triton_kernel_compute[grid](
            input, qweight, scale, output, M, N, K,
            output.stride(0), output.stride(1),
            BLOCK_K=block_size, allow_tf32=bool(torch.backends.cudnn.allow_tf32)
        )

        return output


@triton.autotune(
    configs=[triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128, 'GROUP_M': 8}, num_stages=2, num_warps=8)],
    key=['M', 'N', 'K']
)
@triton.jit
def _triton_kernel_compute(
    input, qweight, scale, output, M, N, K,
    stride_output_m, stride_output_n,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, GROUP_M: tl.constexpr,
    BLOCK_K: tl.constexpr, allow_tf32: tl.constexpr
):

    pid = tl.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    grid_k = tl.cdiv(K, BLOCK_K)

    num_block_in_group = GROUP_M * grid_n
    group_id = pid // num_block_in_group
    group_size_m = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size_m)           # block-num in row
    pid_n = (pid % num_block_in_group) // group_size_m          # block-num in column

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)                # pid row-num
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)                # pid column-num
    rk = tl.arange(0, BLOCK_K)

    input = input + (rm[:, None] * K + rk[None, :])             # (BLOCK_SIZE_M, BLOCK_SIZE_K)
    input_mask = (rm[:, None] < M)
    qweight = qweight + ((rk[:, None] // 2) * N + rn[None, :])  # (BLOCK_SIZE_K, BLOCK_SIZE_N)
    scale = scale + (0 + rn[None, :])                           # (1, BlOCK_SIZE_N)

    B_shift = ((rk % 2) * 4)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # 每计算一个block_m*block_n的output块，都需要加载block_m*K的input、K*block_n的weight、G*block_n的scale
    for k in range(0, grid_k):

        a = tl.load(input, mask=input_mask)
        b = tl.load(qweight)
        b_scale = tl.load(scale)

        b = ((b >> B_shift[:, None]) & 0xF).to(tl.int8)
        b = (b - 0x8) * b_scale

        acc += tl.dot(a, b, allow_tf32=allow_tf32)

        input += BLOCK_K
        qweight += (BLOCK_K // 2) * N
        scale += N

    output = output + (rm[:, None] * stride_output_m + rn[None, :] * stride_output_n)

    mask = (rm < M)[:, None] & (rn < N)[None, :]

    tl.store(output, acc, mask=mask)
