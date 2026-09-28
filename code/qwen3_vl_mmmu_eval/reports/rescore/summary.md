# Rescore summary (orig = Qwen rule parser, v2 = orig + conservative fallback)

| run | acc orig | acc v2 | Δ | trunc% | unparsed% orig | unparsed% v2 | fallback used/correct | v2 unparsed (not truncated) |
|---|---|---|---|---|---|---|---|---|
| baseline_seed42_4k | 44.11 | 58.89 | +14.78 | 19.2 | 37.3 | 16.8 | 185/133 | 18 (open 17) |
| baseline_seed42_8k | 46.56 | 61.56 | +15.00 | 14.1 | 33.6 | 11.3 | 200/135 | 20 (open 20) |
| baseline_seed42_16k | 48.89 | 64.22 | +15.33 | 10.2 | 30.2 | 8.6 | 195/138 | 23 (open 23) |
| baseline_seed42_high_max | 47.00 | 63.00 | +16.00 | 14.2 | 34.6 | 11.2 | 210/144 | 18 (open 18) |
| baseline_seed42_high_min | 45.67 | 62.11 | +16.44 | 12.9 | 35.0 | 11.6 | 211/148 | 22 (open 21) |
| baseline_seed3407_baseline | 47.00 | 62.89 | +15.89 | 12.6 | 31.7 | 9.4 | 200/143 | 20 (open 20) |
| baseline_seed3407_prompt | 59.22 | 59.22 | +0.00 | 14.8 | 13.3 | 13.3 | 0/0 | 20 (open 20) |
| baseline_seed3407_detailed_prompt | 62.00 | 62.11 | +0.11 | 15.2 | 13.8 | 13.2 | 5/1 | 21 (open 21) |
| baseline_seed3407_presence0 | 46.56 | 58.89 | +12.33 | 24.0 | 34.7 | 17.7 | 153/111 | 20 (open 19) |
| final_seed3407_16k | 48.89 | 63.89 | +15.00 | 9.1 | 30.3 | 8.8 | 194/135 | 24 (open 23) |

## McNemar (paired, exact two-sided)

| from → to | parser | gain | loss | p |
|---|---|---|---|---|
| baseline_seed42_8k → baseline_seed42_4k | orig | 69 | 91 | 0.0966 |
| baseline_seed42_8k → baseline_seed42_4k | v2 | 39 | 63 | 0.0223 |
| baseline_seed42_8k → baseline_seed42_16k | orig | 89 | 68 | 0.11 |
| baseline_seed42_8k → baseline_seed42_16k | v2 | 60 | 36 | 0.0184 |
| baseline_seed42_4k → baseline_seed42_16k | orig | 106 | 63 | 0.00117 |
| baseline_seed42_4k → baseline_seed42_16k | v2 | 78 | 30 | 4.31e-06 |
| baseline_seed42_8k → baseline_seed42_high_max | orig | 84 | 80 | 0.815 |
| baseline_seed42_8k → baseline_seed42_high_max | v2 | 61 | 48 | 0.25 |
| baseline_seed42_8k → baseline_seed42_high_min | orig | 98 | 106 | 0.624 |
| baseline_seed42_8k → baseline_seed42_high_min | v2 | 63 | 58 | 0.716 |
| baseline_seed42_8k → baseline_seed3407_baseline | orig | 111 | 107 | 0.839 |
| baseline_seed42_8k → baseline_seed3407_baseline | v2 | 73 | 61 | 0.342 |
| baseline_seed3407_baseline → baseline_seed3407_prompt | orig | 185 | 75 | 6.72e-12 |
| baseline_seed3407_baseline → baseline_seed3407_prompt | v2 | 63 | 96 | 0.0109 |
| baseline_seed3407_baseline → baseline_seed3407_detailed_prompt | orig | 200 | 65 | 3.68e-17 |
| baseline_seed3407_baseline → baseline_seed3407_detailed_prompt | v2 | 73 | 80 | 0.628 |
| baseline_seed3407_baseline → baseline_seed3407_presence0 | orig | 99 | 103 | 0.833 |
| baseline_seed3407_baseline → baseline_seed3407_presence0 | v2 | 40 | 76 | 0.00107 |
| baseline_seed3407_prompt → baseline_seed3407_detailed_prompt | orig | 88 | 63 | 0.0504 |
| baseline_seed3407_prompt → baseline_seed3407_detailed_prompt | v2 | 88 | 62 | 0.0409 |
| baseline_seed3407_baseline → final_seed3407_16k | orig | 93 | 76 | 0.218 |
| baseline_seed3407_baseline → final_seed3407_16k | v2 | 53 | 44 | 0.417 |
| baseline_seed42_16k → final_seed3407_16k | orig | 91 | 91 | 1 |
| baseline_seed42_16k → final_seed3407_16k | v2 | 55 | 58 | 0.851 |
