| GPUs | scenario | ResiHP | no TP shrink | no repartition | no migration | replica drop |
|---|---|---|---|---|---|---|
| 64 | 1 GPU lost | 0.899 | 0.898 | 0.800 | 0.898 | 0.566 |
| 64 | 2 GPUs lost (2 replicas) | 0.910 | stops | 0.568 | 0.910 | 0.317 |
| 64 | node lost | 0.898 | 0.898 | 0.596 | 0.898 | 0.566 |
| 64 | 1 GPU 2x slow | 0.899 | 0.898 | 0.800 | 0.898 | 0.500 |
| 128 | 1 GPU lost | 0.915 | 0.873 | 0.900 | 0.915 | 0.758 |
| 128 | 2 GPUs lost (2 replicas) | 0.871 | 0.854 | 0.790 | 0.821 | 0.698 |
| 128 | node lost | 0.873 | 0.873 | 0.795 | 0.873 | 0.758 |
| 128 | 1 GPU 2x slow | 0.915 | 0.915 | 0.900 | 0.915 | 0.500 |

`replica drop` for a slow GPU: synchronous DP waits for the slow replica (0.5 at 2x).
