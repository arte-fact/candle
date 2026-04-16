# Kernel-family baseline — gemma-4-26B-A4B-it-Q8_0.gguf (candle)

- captured_at: 2026-04-16T11:20:07+00:00
- gpus: 0,1,2,3
- prompt_tokens: 663, gen_tokens: 127
- throughput (no rocprof): {'pp_t_s': 654.82, 'tg_t_s': 46.37}
- cpu_seconds: {'real': 73.275, 'user': 128.579, 'sys': 463.531}

```
family                  calls     total_ms     avg_us     pct  role
--------------------------------------------------------------------------
moe_fused                 120      6080.33   50669.40  60.55%  DEGRADE-TARGET
mmvq_q8                 34050      1477.75      43.40  14.72%  PRESERVE
bookkeeping             63401       753.16      11.88   7.50%  DEGRADE-TARGET
mmq_q8_prefill            410       506.14    1234.48   5.04%  PRESERVE
rmsnorm                 35230       234.82       6.67   2.34%  PRESERVE
elementwise             43030       222.24       5.16   2.21%  DEGRADE-TARGET
router_topk             11700       211.35      18.06   2.10%  DEGRADE-TARGET
flash_decode             7680       195.95      25.51   1.95%  PRESERVE
quant_q8_1              24980       136.49       5.46   1.36%  PRESERVE
rocblas_fallback          120        55.22     460.15   0.55%  
activation               7800        43.75       5.61   0.44%  PRESERVE
softmax_attn             3960        35.46       8.95   0.35%  PRESERVE
rope                     7800        35.45       4.54   0.35%  PRESERVE
sampler_gpu               778        33.58      43.17   0.33%  
moe_combine              3900        19.88       5.10   0.20%  DEGRADE-TARGET
--------------------------------------------------------------------------
TOTAL                             10041.57
```
