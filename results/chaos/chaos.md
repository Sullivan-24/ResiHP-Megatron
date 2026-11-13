| fault | recovered | detect | abort | plan | build + migrate | commit | fault -> commit | commit -> 2 iters | semantics |
|---|---|---|---|---|---|---|---|---|---|
| kill | 3/3 | 0.01s | 0.01s | 0.006s | 0.03s | 0.011s | 0.07s | 0.21s | exact |
| hang | 3/3 | 2.98s | 0.02s | 0.009s | 0.04s | 0.005s | 3.06s | 7.28s | exact |
| slow | 3/3 | 0.18s | 0.06s | 0.005s | 0.05s | 0.007s | 0.31s | 0.00s | exact |
