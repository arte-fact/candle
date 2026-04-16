# Kernel-family baseline — gemma-4-26B-A4B-it-Q8_0.gguf (candle)

- captured_at: 2026-04-16T14:10:01+00:00
- gpus: 0,1,2,3
- prompt_tokens: 663, gen_tokens: 127
- throughput (no rocprof): {'pp_t_s': 677.96, 'tg_t_s': 47.68}
- cpu_seconds: {'real': 68.063, 'user': 119.303, 'sys': 455.082}

```
family                  calls     total_ms     avg_us     pct  role
--------------------------------------------------------------------------
moe_fused                 120      5564.66   46372.15  59.02%  DEGRADE-TARGET
mmvq_q8                 34050      1513.74      44.46  16.06%  PRESERVE
bookkeeping             47801       683.70      14.30   7.25%  DEGRADE-TARGET
mmq_q8_prefill            410       450.39    1098.52   4.78%  PRESERVE
rmsnorm                 35230       238.79       6.78   2.53%  PRESERVE
router_topk             11700       212.83      18.19   2.26%  DEGRADE-TARGET
elementwise             35290       204.62       5.80   2.17%  DEGRADE-TARGET
flash_decode             7680       200.30      26.08   2.12%  PRESERVE
quant_q8_1              21140       125.25       5.92   1.33%  PRESERVE
rocblas_fallback          120        55.09     459.12   0.58%  
rope                     7800        37.08       4.75   0.39%  PRESERVE
softmax_attn             3960        36.59       9.24   0.39%  PRESERVE
sampler_gpu               778        35.50      45.63   0.38%  
activation               3960        25.59       6.46   0.27%  PRESERVE
moe_combine              3900        24.42       6.26   0.26%  DEGRADE-TARGET
moe_gelu_quant           3840        19.64       5.11   0.21%  DEGRADE-TARGET
--------------------------------------------------------------------------
TOTAL                              9428.21
```
