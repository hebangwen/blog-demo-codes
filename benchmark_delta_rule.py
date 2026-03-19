import time
import torch
import torch.nn.functional as F

def l2norm(x: torch.FloatTensor, dim: int = -1, eps: float = 1e-6):
    """Align with FLA library's l2norm implementation."""
    inv_norm = torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)
    return x * inv_norm


def torch_chunk_gated_delta_rule_original(
    query, key, value, g, beta,
    chunk_size=64, initial_state=None,
    output_final_state=False, use_qk_l2norm_in_kernel=False,
):
    """原始实现 - 带有 masked_fill 和 for 循环"""
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)
    
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32)
        for x in (query, key, value, beta, g)
    ]

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    query = F.pad(query, (0, 0, 0, pad_size))
    key = F.pad(key, (0, 0, 0, pad_size))
    value = F.pad(value, (0, 0, 0, pad_size))
    beta = F.pad(beta, (0, pad_size))
    g = F.pad(g, (0, pad_size))
    total_sequence_length = sequence_length + pad_size
    
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale
    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)
    
    query, key, value, k_beta, v_beta = [
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1])
        for x in (query, key, value, k_beta, v_beta)
    ]
    g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)
    
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=0)
    
    g = g.cumsum(dim=-1)
    decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()
    
    # 问题 1: masked_fill 无法并行
    attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0)
    
    # 问题 2: for 循环串行计算
    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    
    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))
    
    last_recurrent_state = (
        torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim).to(value)
        if initial_state is None
        else initial_state.to(value)
    )
    core_attn_out = torch.zeros_like(value)
    
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=1)
    
    for i in range(0, total_sequence_length // chunk_size):
        q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
        attn = (q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]).masked_fill_(mask, 0)
        v_prime = k_cumdecay[:, :, i] @ last_recurrent_state
        v_new = v_i - v_prime
        attn_inter = (q_i * g[:, :, i, :, None].exp()) @ last_recurrent_state
        core_attn_out[:, :, i] = attn_inter + attn @ v_new
        last_recurrent_state = (
            last_recurrent_state * g[:, :, i, -1, None, None].exp()
            + (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(-1, -2) @ v_new
        )

    if not output_final_state:
        last_recurrent_state = None
    
    core_attn_out = core_attn_out.reshape(
        core_attn_out.shape[0], core_attn_out.shape[1], -1, core_attn_out.shape[-1]
    )
    core_attn_out = core_attn_out[:, :, :sequence_length]
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    
    return core_attn_out, last_recurrent_state


def parallel_neumann(A, max_iter=None):
    """O(log n) parallel Neumann iteration for (I - A)^(-1)."""
    chunk_size = A.shape[-1]
    max_iter = max_iter if max_iter is not None else chunk_size.bit_length()
    I = torch.eye(chunk_size, dtype=A.dtype, device=A.device)
    current_sum = I + A
    current_power = A @ A
    
    for _ in range(1, max_iter):
        current_sum = current_sum + current_power @ current_sum
        current_power = current_power @ current_power
    
    return current_sum


def optimized_chunk_gated_delta_rule(
    query, key, value, g, beta,
    chunk_size=64, initial_state=None,
    output_final_state=False, use_qk_l2norm_in_kernel=False,
):
    """优化实现 - 并行 causal mask 和 O(log n) Neumann 迭代"""
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)
    
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32)
        for x in (query, key, value, beta, g)
    ]

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    query = F.pad(query, (0, 0, 0, pad_size))
    key = F.pad(key, (0, 0, 0, pad_size))
    value = F.pad(value, (0, 0, 0, pad_size))
    beta = F.pad(beta, (0, pad_size))
    g = F.pad(g, (0, pad_size))
    total_sequence_length = sequence_length + pad_size
    
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale
    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)
    
    query, key, value, k_beta, v_beta = [
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1])
        for x in (query, key, value, k_beta, v_beta)
    ]
    g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)
    
    # === 优化 1: 构造可并行的因果掩码 ===
    g = g.cumsum(dim=-1)
    g_diff = g.unsqueeze(-1) - g.unsqueeze(-2)
    
    causal_mask_decay = torch.triu(torch.full((chunk_size, chunk_size), -torch.inf, device=query.device))
    identity = torch.eye(chunk_size, device=query.device)
    
    attn_mask = (g_diff + causal_mask_decay).exp()
    attn_mask_qk = attn_mask + identity
    attn = (k_beta @ key.transpose(-1, -2)) * attn_mask
    
    # === 优化 2: O(log n) 并行 Neumann 迭代 ===
    attn = parallel_neumann(attn, max_iter=chunk_size.bit_length())
    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))
    
    last_recurrent_state = (
        torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim).to(value)
        if initial_state is None
        else initial_state.to(value)
    )
    core_attn_out = torch.zeros_like(value)
    
    for i in range(0, total_sequence_length // chunk_size):
        q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
        attn_i = (q_i @ k_i.transpose(-1, -2)) * attn_mask_qk[:, :, i]
        
        v_prime = k_cumdecay[:, :, i] @ last_recurrent_state
        v_new = v_i - v_prime
        attn_inter = (q_i * g[:, :, i, :, None].exp()) @ last_recurrent_state
        core_attn_out[:, :, i] = attn_inter + attn_i @ v_new
        
        last_recurrent_state = (
            last_recurrent_state * g[:, :, i, -1, None, None].exp()
            + (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(-1, -2) @ v_new
        )

    if not output_final_state:
        last_recurrent_state = None
    
    core_attn_out = core_attn_out.reshape(
        core_attn_out.shape[0], core_attn_out.shape[1], -1, core_attn_out.shape[-1]
    )
    core_attn_out = core_attn_out[:, :, :sequence_length]
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    
    return core_attn_out, last_recurrent_state


def benchmark(func, *args, num_runs=50, device='cpu'):
    """Benchmark function with warmup"""
    # Warm up
    for _ in range(5):
        _ = func(*args)
    
    if device == 'cuda':
        torch.cuda.synchronize()
    
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        _ = func(*args)
        if device == 'cuda':
            torch.cuda.synchronize()
        end = time.perf_counter()
        times.append(end - start)
    
    return sum(times) / len(times)


def run_benchmarks():
    """Run benchmarks on different configurations"""
    print("=" * 70)
    print("Chunked Gated Delta Rule Benchmark")
    print("=" * 70)
    
    # Detect device
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        device_name = "Apple Silicon (MPS)"
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        device_name = f"CUDA ({torch.cuda.get_device_name(0)})"
    else:
        device = torch.device('cpu')
        device_name = "CPU"
    
    print(f"\n设备: {device_name}")
    print(f"PyTorch 版本: {torch.__version__}")
    print()
    
    # Test configurations
    configs = [
        {"B": 1, "H": 32, "N": 512,  "D": 256, "chunk": 64},
        {"B": 2, "H": 32, "N": 1024, "D": 256, "chunk": 64},
        {"B": 2, "H": 32, "N": 2048, "D": 256, "chunk": 64},
        {"B": 4, "H": 32, "N": 1024, "D": 256, "chunk": 64},
    ]
    
    results = []
    
    for cfg in configs:
        B, H, N, D, chunk = cfg["B"], cfg["H"], cfg["N"], cfg["D"], cfg["chunk"]
        print(f"\n配置: Batch={B}, Heads={H}, Seq={N}, Dim={D}, Chunk={chunk}")
        print("-" * 70)
        
        # Create test data
        query = torch.randn(B, N, H, D, device=device)
        key = torch.randn(B, N, H, D, device=device)
        value = torch.randn(B, N, H, D, device=device)
        A = torch.empty(H, device=device).uniform_(0, 16)
        bias = torch.ones(H, device=device)
        g = torch.randn(B, N, H, device=device)
        g = -A.float().exp() * F.softplus(g + bias)
        beta = torch.sigmoid(torch.randn(B, N, H, device=device))
        
        # Test correctness first
        with torch.no_grad():
            out_orig, _ = torch_chunk_gated_delta_rule_original(
                query, key, value, g, beta, chunk_size=chunk, use_qk_l2norm_in_kernel=True
            )
            out_opt, _ = optimized_chunk_gated_delta_rule(
                query, key, value, g, beta, chunk_size=chunk, use_qk_l2norm_in_kernel=True
            )
            
            max_diff = (out_orig - out_opt).abs().max().item()
            print(f"数值精度检查: max_diff = {max_diff:.2e}")

        # Benchmark original
        orig_time = benchmark(
            torch_chunk_gated_delta_rule_original, 
            query, key, value, g, beta, chunk,
            device=device.type
        )
        
        # Benchmark optimized
        opt_time = benchmark(
            optimized_chunk_gated_delta_rule,
            query, key, value, g, beta, chunk,
            device=device.type
        )
        
        speedup = orig_time / opt_time
        
        print(f"原始实现: {orig_time*1000:.2f} ms")
        print(f"优化实现: {opt_time*1000:.2f} ms")
        print(f"加速比:   {speedup:.2f}x")
        
        results.append({
            "config": cfg,
            "original_ms": orig_time * 1000,
            "optimized_ms": opt_time * 1000,
            "speedup": speedup,
            "max_diff": max_diff
        })
    
    # Summary
    print("\n" + "=" * 70)
    print("Benchmark Summary")
    print("=" * 70)
    print(f"{'Config':<30} {'Original':>12} {'Optimized':>12} {'Speedup':>10}")
    print("-" * 70)
    for r in results:
        cfg_str = f"B={r['config']['B']}, H={r['config']['H']}, N={r['config']['N']}"
        print(f"{cfg_str:<30} {r['original_ms']:>10.2f}ms {r['optimized_ms']:>10.2f}ms {r['speedup']:>8.2f}x")
    
    avg_speedup = sum(r['speedup'] for r in results) / len(results)
    print("-" * 70)
    print(f"平均加速比: {avg_speedup:.2f}x")
    print("=" * 70)
    
    return results


if __name__ == "__main__":
    results = run_benchmarks()
