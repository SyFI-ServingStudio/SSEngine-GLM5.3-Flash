# Correctness reference (`1_contract.md` §4, §2.1 lengths)

The first output token of each `data/bringup/one_<L>.csv` prompt (8 lengths × 5 requests), served by the vLLM fork as a
single TP4 instance. The PP4 engine must produce the same token ids for the same prompts.

## Build and flags

| | |
|---|---|
| Serving code | vLLM fork `pp-impl/third_party/vllm` at `892da0822f`, in `pp-impl/.venv` (final per `lane-model` 8ec9465): torch 2.13.0+cu130, FlashInfer 0.7.2.dev20261008. See `setup.md`. |
| Model | `nvidia/GLM-5.3-Flash-NVFP4`, snapshot `da920bb0b9f4a06727223a349e55468e38352348` |
| Job | Slurm 7029, 4 × B200 (host GPUs 1, 4, 5, 7), 2026-10-15 03:57–04:20. Server ready after 770 s. KV cache 18,110,005 tokens. |
| Serve command | `python -m vllm.entrypoints.cli.main serve <snapshot> --served-model-name glm53-flash-nvfp4 --tensor-parallel-size 4 --enable-expert-parallel --kv-cache-dtype fp8_e4m3 --max-model-len 1048576 --no-enable-prefix-caching --language-model-only --tokens-only --enable-prompt-tokens-details` (`runs/reference_logs/7029/serve_cmd.txt`) |
| Decoding | Greedy (`temperature 0`), 1 output token, `ignore_eos`. MTP and speculative decoding are off: no `--speculative-config` was passed, and vLLM does not speculate without one. |

## Prompts

req-frontend `7f6aa9d` (`launcher run`, backend `vllm-tokens`, `text-generation-independent` trace format, `trace-timed`)
drives each `one_<L>.csv` through a recording proxy (`bench/reference/record_proxy.py`). The prompts are token ids
taken from the enwik9 token pool: corpus lines are tokenized in order with no special tokens, and request k gets
`pool[k·9973 : k·9973 + L]`. This is deterministic. Before the job, two stub runs plus the Python rebuild
(`bench/reference/make_prompts.py`) gave identical prompts for all 15 checked. In job 7029, every prompt the proxy saw
has the sha256 (int32 token ids) of the rebuild.

Per length, `runs/reference_one_<L>/` holds:
- `run.yaml`, plus `client/` (req-frontend's config, `requests.jsonl` and summary);
- `proxy.jsonl`, what was sent and returned: prompt sha256, first and last 8 ids, sampling params and output ids;
- `prompts.jsonl`, the rebuilt prompt sha256s;
- `logprobs.jsonl`, the reference pass.

The reference pass (`bench/reference/ref_logprobs.py`) sends each rebuilt prompt again, non-streamed, with
`max_tokens 1`, `temperature 0` and the top-5 logprobs. All 40 rows below come from this pass. They are also in
`reference.jsonl`.

## How to compare

1. Rebuild the prompts with `make_prompts.py <one_L.csv> <enwik9> <tokenizer.json> --out DIR`. Check that each
   sha256 equals the table's.
2. Send each prompt to the PP4 engine with greedy decoding and 1 output token.
3. The engine passes a prompt if its token equals the reference token.
4. If the reference's top-2 logprobs are tied (margin 0, here `one_64k_4`), either tied token passes.
5. When the margin is small, as in `one_128k_0` (0.5), a different token means the logits moved slightly. Compare
   the engine's top-5 logprobs with `reference.jsonl` before calling it a failure.

## Result

The reference pass returned a token for all 40 prompts, and every prompt sha256 matches the rebuild. The run pass is
req-frontend through the proxy:
- In 32 requests, its token equals the reference token.
- In `one_64k_4`, it returned 23896 where the reference pass returned 6708. The two have identical logprobs (−0.7334),
  a tie broken differently between the passes.
- In 7 requests, the proxy lost the run record. req-frontend closed the connection after the last event, and the proxy
  then failed to write the record. This was fixed in `28c591a` and tested on a stub; job 7029 ran before the fix. In
  those 7 requests the reference pass alone gives the token. req-frontend counted all 40 requests as SUCCESS with 1
  output token.

| Length | Request | Prompt tokens | Prompt sha256 (first 16) | Reference token | Run token | Top-2 logprobs | Margin | Check |
|---|---|---|---|---|---|---|---|---|
| 1k | `one_1k_0` | 1,024 | `bf4c22dedc655d52` | 29 | 29 | 29 (-0.0001), 1784 (-10.0001) | 10.0 | ok |
| 1k | `one_1k_1` | 1,024 | `1a5850090c06e501` | 609 | 609 | 609 (-0.0454), 3355 (-3.4204) | 3.375 | ok |
| 1k | `one_1k_2` | 1,024 | `1f197fbaf6b2306b` | 1177 | 1177 | 1177 (-0.1763), 5 (-2.8638) | 2.6875 | ok |
| 1k | `one_1k_3` | 1,024 | `7ffc9ea1c16cb1d3` | 2669 | 2669 | 2669 (-0.1892), 11490 (-2.1892) | 2.0 | ok |
| 1k | `one_1k_4` | 1,024 | `659c8cc8eb52f122` | 366 | 366 | 366 (-0.0001), 3500 (-10.8751) | 10.875 | ok |
| 2k | `one_2k_0` | 2,048 | `78e11889ba0707db` | 2359 | 2359 | 2359 (-0.3671), 1291 (-2.3671) | 2.0 | ok |
| 2k | `one_2k_1` | 2,048 | `901606b59ed50549` | 36040 | 36040 | 36040 (-0.4359), 1425 (-1.4359) | 1.0 | ok |
| 2k | `one_2k_2` | 2,048 | `be952cb4e6ac7142` | 30084 | 30084 | 30084 (-0.0683), 729 (-3.4433) | 3.375 | ok |
| 2k | `one_2k_3` | 2,048 | `0d2723925bf67155` | 72219 | 72219 | 72219 (-0.0028), 422 (-6.6903) | 6.6875 | ok |
| 2k | `one_2k_4` | 2,048 | `f5f9db248c8cf4e4` | 91 | 91 | 91 (-0.0383), 50521 (-3.4133) | 3.375 | ok |
| 4k | `one_4k_0` | 4,096 | `fc994a6d07462adc` | 264 | 264 | 264 (-0.0184), 4318 (-5.6434) | 5.625 | ok |
| 4k | `one_4k_1` | 4,096 | `cc74d949733b14f8` | 705 | 705 | 705 (-0.0185), 4318 (-5.2685) | 5.25 | ok |
| 4k | `one_4k_2` | 4,096 | `f51c0556cc03b186` | 323 | 323 | 323 (-0.0023), 11 (-7.9398) | 7.9375 | ok |
| 4k | `one_4k_3` | 4,096 | `e0808e25b1593a6c` | 2101 | 2101 | 2101 (-0.0019), 969 (-7.5019) | 7.5 | ok |
| 4k | `one_4k_4` | 4,096 | `5582c2e6da7e21d4` | 6238 | 6238 | 6238 (-0.0001), 2530 (-10.0001) | 10.0 | ok |
| 8k | `one_8k_0` | 8,192 | `960da58158f6eaa2` | 1352 | 1352 | 1352 (-0.1315), 387 (-3.944) | 3.8125 | ok |
| 8k | `one_8k_1` | 8,192 | `6229e679dd0a250f` | 24836 | 24836 | 24836 (-0.0221), 15368 (-4.7721) | 4.75 | ok |
| 8k | `one_8k_2` | 8,192 | `2e8ab18302b6d179` | 5053 | 5053 | 5053 (-0.0107), 27626 (-4.6357) | 4.625 | ok |
| 8k | `one_8k_3` | 8,192 | `1154e51c66e437d2` | 262 | 262 | 262 (-0.0002), 220 (-9.5627) | 9.5625 | ok |
| 8k | `one_8k_4` | 8,192 | `2469f66c63b98bcd` | 1584 | 1584 | 1584 (-0.0029), 3234 (-6.7529) | 6.75 | ok |
| 16k | `one_16k_0` | 16,384 | `14dd65ea1789b8a7` | 29 | 29 | 29 (-0.0), 15349 (-13.25) | 13.25 | ok |
| 16k | `one_16k_1` | 16,384 | `93e68de6f3cfd696` | 100562 | 100562 | 100562 (-0.0009), 100809 (-8.1259) | 8.125 | ok |
| 16k | `one_16k_2` | 16,384 | `4af971cd1783c80b` | 262 | — | 262 (-0.0006), 220 (-7.6881) | 7.6875 | run record lost (proxy) |
| 16k | `one_16k_3` | 16,384 | `5d6ac9a9e8b24a2e` | 220 | — | 220 (-0.044), 2410 (-3.294) | 3.25 | run record lost (proxy) |
| 16k | `one_16k_4` | 16,384 | `1700c647e3b7c010` | 5053 | 5053 | 5053 (-0.1315), 20449 (-2.1315) | 2.0 | ok |
| 32k | `one_32k_0` | 32,768 | `cd1807d9ee00287f` | 315 | 315 | 315 (-0.0001), 1055 (-10.0001) | 10.0 | ok |
| 32k | `one_32k_1` | 32,768 | `6e0da8959a88a520` | 264 | — | 264 (-0.0612), 362 (-3.4362) | 3.375 | run record lost (proxy) |
| 32k | `one_32k_2` | 32,768 | `da2d266a2abd5e52` | 758 | 758 | 758 (-0.1441), 220 (-2.7691) | 2.625 | ok |
| 32k | `one_32k_3` | 32,768 | `c3c86fc0fffe914e` | 553 | 553 | 553 (-0.2219), 13 (-1.7219) | 1.5 | ok |
| 32k | `one_32k_4` | 32,768 | `d0cf29edb7318e80` | 1030 | 1030 | 1030 (-0.0095), 3381 (-5.072) | 5.0625 | ok |
| 64k | `one_64k_0` | 65,536 | `fcaa96109a3a8e9c` | 1330 | — | 1330 (-0.0), 2840 (-12.0) | 12.0 | run record lost (proxy) |
| 64k | `one_64k_1` | 65,536 | `aa1fea63e2f24d96` | 14 | 14 | 14 (-0.0009), 16327 (-8.3134) | 8.3125 | ok |
| 64k | `one_64k_2` | 65,536 | `8c5c8de30eaa4318` | 269 | 269 | 269 (-0.0002), 813 (-10.0002) | 10.0 | ok |
| 64k | `one_64k_3` | 65,536 | `2023436d47ab5f6d` | 476 | 476 | 476 (-0.1601), 6168 (-2.7851) | 2.625 | ok |
| 64k | `one_64k_4` | 65,536 | `77d316c19e767e4a` | 6708 | 23896 | 6708 (-0.7334), 23896 (-0.7334) | 0.0 | ok (tied top-1) |
| 128k | `one_128k_0` | 131,072 | `099d0236eb921efc` | 4318 | — | 4318 (-0.5251), 96323 (-1.0251) | 0.5 | run record lost (proxy) |
| 128k | `one_128k_1` | 131,072 | `747337b0452403ca` | 50484 | 50484 | 50484 (-0.1268), 96800 (-2.7518) | 2.625 | ok |
| 128k | `one_128k_2` | 131,072 | `2e43821c5ba09f68` | 4318 | — | 4318 (-0.0586), 33866 (-2.9336) | 2.875 | run record lost (proxy) |
| 128k | `one_128k_3` | 131,072 | `ed22367f5b6a4ee1` | 220 | — | 220 (-0.039), 4318 (-3.289) | 3.25 | run record lost (proxy) |
| 128k | `one_128k_4` | 131,072 | `6a68c4ff2efea7c6` | 1075 | 1075 | 1075 (-0.0165), 1741 (-5.2665) | 5.25 | ok |
