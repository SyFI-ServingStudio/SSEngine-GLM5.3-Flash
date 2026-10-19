"""Trace-level k1 extra prefill (ideal cache, k1 state only at the previous decode end), % of all new tokens.
A round whose prompt end L = p + i is below the previous round's decode end recomputes its whole prefix p.
usage: trace_k1_split.py <trace.csv>"""
import sys, duckdb
c = duckdb.connect()
c.sql(f"""create table t as select p+i L, p, i, lag(prefix_len+input_len+output_len-1) over w prev_end, lag(prefix_len+input_len) over w prevL
  from (select *, prefix_len p, input_len i from read_csv_auto('{sys.argv[1]}')) window w as (partition by session_id order by round_idx)""")
new = c.sql("select sum(i) from t").fetchone()[0]
tot = 0
for name, cond in [('inside_output (prevL <= L < prev decode end)', 'L >= prevL and L < prev_end'),
                   ('inside_prompt, drop < 64k', 'L < prevL and prevL - L < 64000'),
                   ('inside_prompt, drop >= 64k', 'L < prevL and prevL - L >= 64000')]:
    n, e, d = c.sql(f"select count(*), sum(p), median(prev_end - L) from t where prev_end is not null and {cond}").fetchone()
    tot += e
    print(f"{name:46s} rounds {n:6d} extra {100*e/new:5.2f}% of new, median shortfall to prev decode end {d}")
print(f"total {100*tot/new:.2f}% of new ({new} new tokens)")
