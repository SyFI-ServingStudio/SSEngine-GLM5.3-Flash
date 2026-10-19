"""Sim target windows for the vLLM comparison: per simulator run dir, rounds/s, new and computed tok/s, hit share of
declared prefix, TTFT p50/p90/p99 (s) and rounds > 60 s over first token in [0, 20) and [10, 20) min (raw/request_slo.parquet).
    <python with duckdb> bench/vllm_dpep_simwin.py <sim run dir> ...
"""
import duckdb, sys, json
c = duckdb.connect()
for run in sys.argv[1:]:
    out = {}
    for name, t0 in (("0_20", 0), ("10_20", 600e3)):
        q = c.sql(f"""select count(*), sum(fresh_prompt_tokens), sum(declared_prefix_tokens + fresh_prompt_tokens - coalesce(prefix_cache_hit_tokens,0)),
          sum(coalesce(prefix_cache_hit_tokens,0)), sum(declared_prefix_tokens),
          quantile_cont(ttft_ms,0.5), quantile_cont(ttft_ms,0.9), quantile_cont(ttft_ms,0.99), sum((ttft_ms>60e3)::int)
          from '{run}/raw/request_slo.parquet' where completed and arrival_time_ms + ttft_ms >= {t0} and arrival_time_ms + ttft_ms < 1200e3""").fetchone()
        n, new, comp, hit, pre, p50, p90, p99, o60 = q
        W = (1200e3 - t0) / 1e3
        out[name] = dict(rps=round(n / W, 2), new_tok_s=round(new / W), comp_tok_s=round(comp / W), hit=round(hit / max(pre, 1), 4),
                         ttft=[round(p50 / 1e3, 2), round(p90 / 1e3, 2), round(p99 / 1e3, 2)], over60=o60)
    print(run.split("logs/")[-1], json.dumps(out))
