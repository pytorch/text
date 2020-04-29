import torch
from torchtext.modules import MultiheadAttentionContainer, ScaledDotProduct
from torch.nn.functional import multi_head_attention_forward as mha_forward
import time


def benchmark_mha_block():

    def _run_benchmark(embed_dim, nhead, tgt_len, src_len, bsz, device):
        # Build torchtext MultiheadAttention module
        MHA = MultiheadAttentionContainer(nhead,
                                          (torch.nn.Linear(embed_dim, embed_dim),
                                           torch.nn.Linear(embed_dim, embed_dim),
                                           torch.nn.Linear(embed_dim, embed_dim),),
                                          ScaledDotProduct(),
                                          torch.nn.Linear(embed_dim, embed_dim)).to(device)

        query = torch.rand((tgt_len, bsz, embed_dim)).to(device)
        key = value = torch.rand((src_len, bsz, embed_dim)).to(device)
        attn_mask_2D = torch.randint(0, 2, (tgt_len, src_len)).to(torch.bool).to(device)
        bias_k = bias_v = torch.rand((1, 1, embed_dim)).to(device)
        print("starting torchtext.modules.MultiheadAttentionContainer")
        t0 = time.monotonic()
        for _ in range(100):
            mha_output, attn_weights = MHA(query, key, value,
                                           attn_mask=torch.stack([attn_mask_2D] * (bsz * nhead)),
                                           bias_k=bias_k.repeat(1, bsz, 1).reshape(1, bsz * nhead, -1),
                                           bias_v=bias_v.repeat(1, bsz, 1).reshape(1, bsz * nhead, -1))
        print(time.monotonic() - t0)

        # Use torch.nn.functional.multi_head_attention_forward
        torch_attn_mask = torch.zeros((tgt_len, src_len)).to(device).masked_fill_(attn_mask_2D, float('-inf'))
        print("starting torch.nn.functional.multi_head_attention_forward")
        in_proj_weight = torch.cat([MHA.query_in_proj.weight, MHA.key_in_proj.weight, MHA.value_in_proj.weight])
        t0 = time.monotonic()
        for _ in range(100):
            torch_mha_output, torch_mha_weights = mha_forward(query, key, value,
                                                              embed_dim, nhead,
                                                              in_proj_weight, None,
                                                              bias_k, bias_v,
                                                              False, 0.0,
                                                              MHA.out_proj.weight,
                                                              MHA.out_proj.bias,
                                                              attn_mask=torch_attn_mask)
        print(time.monotonic() - t0)

    print("*" * 80)
    print("test case GPU with embed_dim, nhead, tgt_len, src_len, bsz:", 768, 12, 128, 128, 72)
    _run_benchmark(768, 12, 128, 128, 72, torch.device("cuda"))

    print("*" * 80)
    print("test case GPU with embed_dim, nhead, tgt_len, src_len, bsz:", 64, 2, 10, 10, 8)
    _run_benchmark(64, 2, 10, 10, 8, torch.device("cuda"))

    print("*" * 80)
    print("test case CPU with embed_dim, nhead, tgt_len, src_len, bsz:", 768, 12, 128, 128, 72)
    _run_benchmark(768, 12, 128, 128, 72, torch.device("cpu"))

    print("*" * 80)
    print("test case CPU with embed_dim, nhead, tgt_len, src_len, bsz:", 64, 2, 10, 10, 8)
    _run_benchmark(64, 2, 10, 10, 8, torch.device("cpu"))


if __name__ == "__main__":
    benchmark_mha_block()
