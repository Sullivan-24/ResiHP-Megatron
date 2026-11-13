from resihp.bench.megatron_log import parse_log, summarize

LOG = """
 [2026-11-06 10:00:00.000000] iteration        1/     200 | consumed samples:          128 | elapsed time per iteration (ms): 9000.1 | throughput per GPU (TFLOP/s/GPU): 100.0 | learning rate: 7.500000E-06 | global batch size:   128 | lm loss: 1.180000E+01 |
 [2026-11-06 10:00:05.000000] iteration        2/     200 | consumed samples:          256 | elapsed time per iteration (ms): 5000.0 | throughput per GPU (TFLOP/s/GPU): 410.2 | learning rate: 1.500000E-05 | global batch size:   128 | lm loss: 1.170000E+01 |
 [2026-11-06 10:00:10.000000] iteration        3/     200 | consumed samples:          384 | elapsed time per iteration (ms): 5002.0 | throughput per GPU (TFLOP/s/GPU): 409.8 | learning rate: 2.250000E-05 | global batch size:   128 | lm loss: 1.150000E+01 |
 some unrelated line
"""


def test_parse_and_summarize():
    recs = parse_log(LOG)
    assert [r.iteration for r in recs] == [1, 2, 3]
    assert recs[1].tflops_per_gpu == 410.2 and recs[2].lm_loss == 11.5 and recs[0].global_batch_size == 128
    s = summarize(recs, skip_first=1)
    assert s["median_iter_time_ms"] == 5001.0
    assert s["final_loss"] == 11.5
