# Rescore summary (orig = Qwen rule parser, v2 = orig + conservative fallback)

| run | acc orig | acc v2 | Δ | trunc% | unparsed% orig | unparsed% v2 | fallback used/correct | v2 unparsed (not truncated) |
|---|---|---|---|---|---|---|---|---|
| lora_merged | 50.00 | 66.22 | +16.22 | 2.4 | 27.0 | 3.9 | 208/146 | 25 (open 25) |

## McNemar (paired, exact two-sided)

| from → to | parser | gain | loss | p |
|---|---|---|---|---|
