# Kernel-family baseline — gemma-4-26B-A4B-it-Q8_0.gguf (candle)

- captured_at: 2026-04-16T08:28:34+00:00
- gpus: 0,1,2,3
- prompt_tokens: 663, gen_tokens: 127
- throughput (no rocprof): {'pp_t_s': 566.11, 'tg_t_s': 43.84}
- throughput (under rocprof): {'pp_t_s': 198.45, 'tg_t_s': 26.13}
- cpu_seconds: {'real': 67.531, 'user': 122.596, 'sys': 282.418}

```
family                  calls     total_ms     avg_us     pct  role
--------------------------------------------------------------------------
moe_fused                7800      6126.95     785.51  61.45%  DEGRADE-TARGET
mmvq_q8                 26370       916.91      34.77   9.20%  PRESERVE
bookkeeping             71201       799.44      11.23   8.02%  DEGRADE-TARGET
router_topk             15600       541.76      34.73   5.43%  DEGRADE-TARGET
mmq_q8_prefill            410       532.65    1299.14   5.34%  PRESERVE
elementwise             46930       273.70       5.83   2.75%  DEGRADE-TARGET
rmsnorm                 35230       244.09       6.93   2.45%  PRESERVE
flash_decode             7680       197.55      25.72   1.98%  PRESERVE
quant_q8_1              24980       135.24       5.41   1.36%  PRESERVE
rocblas_fallback          120        54.68     455.67   0.55%  
activation               7800        42.57       5.46   0.43%  PRESERVE
rope                     7800        35.95       4.61   0.36%  PRESERVE
softmax_attn             3960        35.81       9.04   0.36%  PRESERVE
sampler_gpu               778        33.49      43.05   0.34%  
--------------------------------------------------------------------------
TOTAL                              9970.78
```
