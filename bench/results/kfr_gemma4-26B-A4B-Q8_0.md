# Kernel-family baseline — gemma-4-26B-A4B-it-Q8_0.gguf (candle)

- captured_at: 2026-04-16T13:30:12+00:00
- gpus: 0,1,2,3
- prompt_tokens: 663, gen_tokens: 127
- throughput (no rocprof): {'pp_t_s': 675.96, 'tg_t_s': 47.65}
- cpu_seconds: {'real': 71.164, 'user': 126.717, 'sys': 476.279}

```
family                  calls     total_ms     avg_us     pct  role
--------------------------------------------------------------------------
moe_fused                 120      5571.11   46425.88  58.36%  DEGRADE-TARGET
mmvq_q8                 34050      1463.97      42.99  15.33%  PRESERVE
bookkeeping             48041       679.78      14.15   7.12%  DEGRADE-TARGET
mmq_q8_prefill            410       448.94    1094.98   4.70%  PRESERVE
rocblas_fallback          120       241.22    2010.17   2.53%  
rmsnorm                 35230       236.36       6.71   2.48%  PRESERVE
elementwise             39190       210.07       5.36   2.20%  DEGRADE-TARGET
router_topk             11700       207.23      17.71   2.17%  DEGRADE-TARGET
flash_decode             7680       195.69      25.48   2.05%  PRESERVE
quant_q8_1              21140       118.89       5.62   1.25%  PRESERVE
rope                     7800        36.12       4.63   0.38%  PRESERVE
sampler_gpu               778        35.16      45.19   0.37%  
softmax_attn             3960        34.66       8.75   0.36%  PRESERVE
activation               3960        24.63       6.22   0.26%  PRESERVE
moe_combine              3900        23.71       6.08   0.25%  DEGRADE-TARGET
moe_gelu_quant           3840        19.08       4.97   0.20%  DEGRADE-TARGET
--------------------------------------------------------------------------
TOTAL                              9546.61
```
