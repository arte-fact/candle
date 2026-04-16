# Kernel-family baseline — gemma-4-26B-A4B-it-Q8_0.gguf (turbo)

- captured_at: 2026-04-16T08:28:40+00:00
- gpus: 0,1,2,3
- prompt_tokens: 512, gen_tokens: 128
- throughput (no rocprof): {'pp_t_s': 899.46, 'tg_t_s': 66.74}
- throughput (under rocprof): {'pp_t_s': 890.66, 'tg_t_s': 41.25}
- cpu_seconds: {'real': 29.722, 'user': 45.745, 'sys': 9.405}

```
family                  calls     total_ms     avg_us     pct  role
--------------------------------------------------------------------------
mmvq_q8                 45938      1404.60      30.58  42.02%  PRESERVE
mmq_q8_prefill            520       925.27    1779.36  27.68%  PRESERVE
rmsnorm                 43361       277.33       6.40   8.30%  PRESERVE
quant_q8_1              34846       175.60       5.04   5.25%  PRESERVE
elementwise             27903       167.68       6.01   5.02%  DEGRADE-TARGET
rocblas_fallback          178        75.92     426.49   2.27%  
kv_index                12052        73.13       6.07   2.19%  
bookkeeping              9845        65.21       6.62   1.95%  DEGRADE-TARGET
router_topk              4046        54.47      13.46   1.63%  DEGRADE-TARGET
rope                     7860        48.43       6.16   1.45%  PRESERVE
activation               7860        44.78       5.70   1.34%  PRESERVE
softmax_attn             3930        30.08       7.65   0.90%  PRESERVE
--------------------------------------------------------------------------
TOTAL                              3342.49
```
