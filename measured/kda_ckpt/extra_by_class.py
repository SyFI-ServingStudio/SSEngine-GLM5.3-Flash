"""Per-class extra compute (computed - new tokens) as % of new tokens, rounds with first token in [1 h, 8 h).
usage: extra_by_class.py <run dir>=<trace> ..."""
import sys, duckdb
t0, t1 = 3.6e6, 2.88e7
c = duckdb.connect()
for arg in sys.argv[1:]:
    run, trace = arg.split('=')
    c.sql(f"create or replace table tr as select row_number() over () - 1 as rid, session_id, round_idx, prefix_len p, input_len i, output_len o from read_csv('{trace}')")
    c.sql(f"create or replace table r as select request_id rid, completed, arrival_time_ms arr, ttft_ms, declared_prefix_tokens pre, fresh_prompt_tokens nw, coalesce(prefix_cache_hit_tokens,0) hit from '{run}/raw/request_slo.parquet'")
    c.sql("""create or replace table y as select *, case when prev_end is null then 'cold' when pre+nw <= prev_prompt_end then 'inside_prompt'
      when pre+nw <= prev_end then 'inside_output' else 'extend' end cls from (select r.*, lag(tr.p+tr.i) over w prev_prompt_end,
      lag(tr.p+tr.i+greatest(tr.o,1)-1) over w prev_end from r join tr using (rid) window w as (partition by tr.session_id order by tr.round_idx))""")
    rows = c.sql(f"""select cls, count(*), sum(pre+nw-hit-nw) from y where completed and arr+ttft_ms >= {t0} and arr+ttft_ms < {t1} group by cls""").fetchall()
    tot = c.sql(f"select count(*), sum(nw), sum(pre+nw-hit-nw) from y where completed and arr+ttft_ms >= {t0} and arr+ttft_ms < {t1}").fetchone()
    n, new, extra = tot
    d = {k: (cnt, e) for k, cnt, e in rows}
    print(run.split('logs/')[-1], f"rounds={n} extra/new={100*extra/new:.2f}%", ' '.join(f"{k}:{d[k][0]}r/{100*d[k][1]/new:.2f}%" for k in ['inside_output','inside_prompt','extend','cold'] if k in d))
