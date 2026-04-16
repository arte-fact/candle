# Kernel-family baseline — gemma-4-26B-A4B-it-Q8_0.gguf (candle)

- captured_at: 2026-04-16T12:53:18+00:00
- gpus: 0,1,2,3
- prompt_tokens: 663, gen_tokens: 127
- throughput (no rocprof): {'pp_t_s': 653.66, 'tg_t_s': 47.45}
- cpu_seconds: {'real': 68.203, 'user': 116.828, 'sys': 457.622}

```
family                  calls     total_ms     avg_us     pct  role
--------------------------------------------------------------------------
moe_fused                 120      5564.39   46369.90  59.12%  DEGRADE-TARGET
mmvq_q8                 34050      1498.36      44.00  15.92%  PRESERVE
bookkeeping             48041       684.64      14.25   7.27%  DEGRADE-TARGET
mmq_q8_prefill            410       449.86    1097.22   4.78%  PRESERVE
rmsnorm                 35230       238.22       6.76   2.53%  PRESERVE
elementwise             39190       216.50       5.52   2.30%  DEGRADE-TARGET
router_topk             11700       210.45      17.99   2.24%  DEGRADE-TARGET
flash_decode             7680       198.93      25.90   2.11%  PRESERVE
quant_q8_1              21140       120.19       5.69   1.28%  PRESERVE
rocblas_fallback          120        55.03     458.57   0.58%  
rope                     7800        36.31       4.66   0.39%  PRESERVE
softmax_attn             3960        35.55       8.98   0.38%  PRESERVE
sampler_gpu               778        34.35      44.15   0.36%  
activation               3960        25.12       6.34   0.27%  PRESERVE
moe_combine              3900        24.12       6.19   0.26%  DEGRADE-TARGET
moe_gelu_quant           3840        19.72       5.14   0.21%  DEGRADE-TARGET
--------------------------------------------------------------------------
TOTAL                              9411.73
```
