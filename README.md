# Ling-3.1-flash-1019 on 2× DGX Spark (GB10) — our TensorFold TP2 engine

**Bertholomus AI recipe: inclusionAI's Ling-3.1-flash-1019 served tensor-parallel over two NVIDIA DGX Spark / GB10
nodes by the model family we wrote for [TensorFold](https://github.com/ashhart/TensorFold) (by ashhart), with the
262,144-token window of the model's config and decoding drafted by the model's own MTP layer.**

> **Weights:** our EXL3 conversion (3.03 bits per weight) of
> [inclusionAI/Ling-3.1-flash-1019](https://huggingface.co/inclusionAI/Ling-3.1-flash-1019) (MIT) at revision
> `dddfee4b`, made with the recipe in [`conversion/`](conversion/), is on Hugging Face as
> [bertholomus/Ling-3.1-flash-1019-EXL3-3.03bpw](https://huggingface.co/bertholomus/Ling-3.1-flash-1019-EXL3-3.03bpw) (220.6 GB).
> [Get the weights](#get-the-weights) downloads it; [Make the weights](#make-the-weights) makes it from the original.

**Engine:** commit `72f16e48` on
[bertholomus/TensorFold, branch `ling-3.1-tp2`](https://github.com/bertholomus/TensorFold/tree/ling-3.1-tp2):
Bertholomus AI's `bailing_v3` family on upstream TensorFold v0.6.6's Python CUDA engine, on top of our GLM-5.3 engine
line (`glm-dsa-tp4`). It holds the code we serve and measured; only our internal notes, paths and one internal script
are taken out. The model math follows inclusionAI's modeling code for Ling-3.1-flash (`modeling_bailing_moe_v3.py`,
Apache-2.0).

Ling-3.1-flash-1019 is a later checkpoint of Ling-3.1-flash with the same architecture: every tensor has the same name,
dtype and shape. `BailingMoeV3ForCausalLM` has 72 layers: 63 with KDA linear attention and 9 with gated MLA (layers 7,
15, …, 71); 4 dense MLP layers, then 68 mixture-of-experts layers with 512 routed experts (8 active) and a shared
expert; and one MTP layer. Its model card gives 560B total and 25B active parameters.

## v0.2.0 (2026-10-28): Ling-3.1-flash-1019 with its native 262K window

Engine `72f16e48` with `--context 262144`: the model's native 262,144-token window, MTP drafting on (the default),
one request at a time, on our 3.03 bpw conversion of the 1019 checkpoint (made as v0.1.0's was, with the same
widths). Two memory settings new in this engine, `TF_LING_SHARED_SCRATCH=1` and `TF_LING_EMBED_SPLIT=1`, free about
1.26 GiB a node, so the larger window keeps more memory free than v0.1.0's 192K did; neither changes the output.
A reply that thinks keeps part of `max_tokens` for its answer ([Thinking and max_tokens](#thinking-and-max_tokens)).

Measured on two GB10 nodes (TP2) joined by a direct RoCE link, at the settings in [Run it](#run-it), with our own
client: greedy (temperature 0), one run per row unless noted.

Drafting does not change the reply: drafted and serial decoding give the same tokens on fresh prompts (6 of 6 test
prompts, and the replies that decode across the cache's 512-key chunk boundaries at every draft depth).

**Decode** (tok/s after the first token):

| Prompt tokens | Reply tokens | Drafted (3 runs) | Drafts accepted | Serial (2 runs) |
|---|---|---|---|---|
| 72 (thinking off) | 256 | 35.1-36.9 | 128-137 / 209-216 | 23.9-24.1 |
| 2,222 | 60-88 | 37.7-38.7 | 36 / 47, 54 / 70 | 22.9-23.0 |
| 6,623 | 60-94 | 36.9-39.3 | 36 / 44, 58 / 77 | 22.5-23.6 |
| 255,112 | 189 | 28.8-29.1 | 119 / 163 | |

How fast drafted decoding runs depends on how many of the MTP layer's drafts the model accepts, which depends on the
text. Sampled requests are drafted too (the server's defaults are temperature 1.0, top-k 20, top-p 0.95).

**Prompt** (time to the first token):

| Prompt tokens | Time | tok/s |
|---|---|---|
| 2,222 | 3.43-3.62 s | 614-648 |
| 2,245 | 3.53 s | 638 |
| 6,623 | 9.52-9.60 s | 690-696 |
| 9,002 | 13.6 s | 662 |
| 36,531 | 64.3 s | 568 |
| 255,112 | 1,137 s (19.0 min) | 224 |

A prompt that extends the previous request's prompt reuses it: the 255,112-token prompt sent again started in 0.10 s.

**Memory** (each node has 123.79 GiB, shared by its CPU and GPU):

| | Rank 0 | Rank 1 |
|---|---|---|
| GPU memory, weights loaded (PyTorch allocated) | 103.41 GiB | 103.41 GiB |
| GPU memory, ready to serve (buffers, caches, graphs) | 107.96 GiB | 107.96 GiB |
| Free (MemAvailable) after the weights load | 9.21 GiB | 8.31 GiB |
| Free when ready to serve | 4.29 GiB | 2.94 GiB |
| Lowest free while serving (the checks in this README, 30 minutes) | 2.11 GiB | 1.94 GiB |

Start to ready: 237 s, with the weights read from each node's local disk.

**Quality: the engine's arithmetic.** As for v0.1.0, we compared the engine's next-token logits at every position of
the same 40 prompts (66,103 positions) with a reference forward of the same EXL3 weights, run once in fp32 throughout
and once with bf16 activations:

| | Top-1 agreement | Mean KL (reference ‖ tested) | Max KL |
|---|---|---|---|
| Engine vs the fp32 reference | 0.9712 | 0.0132 | 8.44 |
| bf16 reference vs the fp32 reference | 0.9726 | 0.0136 | 9.94 |
| Engine vs the bf16 reference | 0.9715 | 0.0138 | 12.35 |

So the engine is as close to exact fp32 arithmetic as the same math with bf16 activations: its KL is 0.97 times as
large, and its top-1 agreement is 0.0013 lower. These numbers measure arithmetic on the same quantized weights. How
far the 3.03 bpw conversion is from the original FP8 model was not measured end to end.

## Earlier releases

v0.1.0 (2026-10-25) served the original Ling-3.1-flash (revision `7b864daa`, our conversion
[bertholomus/Ling-3.1-flash-EXL3-3.03bpw](https://huggingface.co/bertholomus/Ling-3.1-flash-EXL3-3.03bpw)) with a 192K
window on engine `ef914199`, and v0.1.1 (2026-10-28) added the answer reserve (engine `0441f2ac`). Their notes and
numbers are in the README at the tags `v0.1.0` and `v0.1.1`.

## Get the weights

Our conversion on Hugging Face is the folder our served rank folders were split from: steps 1 to 4 below are done,
and its `README.md` is our model card. Download it and the engine, then check and split the weights as in step 5:

```
hf download bertholomus/Ling-3.1-flash-1019-EXL3-3.03bpw --local-dir <EXL3_DIR>
git clone -b ling-3.1-tp2 https://github.com/bertholomus/TensorFold <TENSORFOLD>
```

## Make the weights

1. Download [inclusionAI/Ling-3.1-flash-1019](https://huggingface.co/inclusionAI/Ling-3.1-flash-1019) at revision
   `dddfee4bb43519a92761fbc5cefe3f0d3ff4aded` (FP8, 535 GiB):

   ```
   hf download inclusionAI/Ling-3.1-flash-1019 --revision dddfee4bb43519a92761fbc5cefe3f0d3ff4aded --local-dir <SRC>
   ```

2. Write a BF16 copy for the converter to read (1,066 GiB; about 4 h on one GB10's CPU), and check it, with the
   engine's tools:

   ```
   git clone -b ling-3.1-tp2 https://github.com/bertholomus/TensorFold <TENSORFOLD>
   python3 <TENSORFOLD>/tools/ling_fp8_write_bf16.py <SRC> <BF16_DIR>
   python3 <TENSORFOLD>/tools/ling_bf16_check.py <SRC> <BF16_DIR>
   ```

   Each FP8 weight is multiplied by its block scale in fp32 and rounded once to bf16, and every other tensor is
   copied byte for byte; the scales are dropped, and `config.json` loses its `quantization_config`. The check opens
   every shard, counts the tensors, checks the index, and compares 64 random tensors and each shard's first and last
   with the source bit for bit.

3. Convert the BF16 copy with our exllamav3 branch
   [`ling-3.1`](https://github.com/bertholomus/exllamav3/tree/ling-3.1) (commit `13463cc9`: exllamav3 v1.5.3 plus
   the Ling-3.1 module), in `nvcr.io/nvidia/pytorch:26.07-py3`, with this repository checked out at `<THIS_REPO>`:

   ```
   git clone -b ling-3.1 https://github.com/bertholomus/exllamav3 <EXLLAMAV3>
   cd <EXLLAMAV3>
   TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0 HF_HUB_OFFLINE=1 \
   python convert.py -i <BF16_DIR> -w <WORK> -o <EXL3_DIR> -rcp <THIS_REPO>/conversion/ling-tp2.yaml -hb 6 -mb 4
   ```

   Widths: routed experts 3, shared experts 5, KDA attention 4, dense MLP 4, MLA attention 6 (its `kv_b_proj` stays
   unquantized), output head 6, MTP layer 4; 3.03 bpw by the converter's count, with its default calibration (250
   rows of 2,048 tokens). NVIDIA's container sets `TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1`, which runs fp32 matrix
   products in TF32; our conversion ran without it, and the Ling module refuses to start while it is set. On one
   GB10 a conversion takes about 29 hours (29 h 14 min for the earlier checkpoint) and writes 220.6 GB in 35 shards.
   Ours of 1019 took 20 h 7 min with a second GB10 quantizing part of each layer for the first, through our own
   converter code, which a byte-for-byte check showed writes the same tensors as one node alone.

4. Put back the tensors the converter leaves unquantized, exactly as the BF16 copy has them:

   ```
   python3 <TENSORFOLD>/tools/ling_restore_passthrough.py <BF16_DIR> <EXL3_DIR>
   ```

   The converter stores those tensors (norms, the embedding, the router, `kv_b_proj`, the KDA's short convolutions,
   …) in its own dtype, some of them as F16. The restore writes back the BF16 copy's bytes and dtype, the three
   short convolutions as the one fused tensor the converter writes, and exits 0 only if every restored tensor reads
   back equal. On our conversion it restored 646 tensors.

5. Check the split, then split the weights into one folder per rank (~112 GB each), with the engine's `src/` on
   `PYTHONPATH`:

   ```
   python3 <TENSORFOLD>/tools/ling_split_check.py <EXL3_DIR> --world 2
   TF_TP_WORLD=2 PYTHONPATH=<TENSORFOLD>/src python3 -m tensorfold.families.glm5_next.cuda.split <EXL3_DIR> --rank 0 <RANK_DIR_0>
   TF_TP_WORLD=2 PYTHONPATH=<TENSORFOLD>/src python3 -m tensorfold.families.glm5_next.cuda.split <EXL3_DIR> --rank 1 <RANK_DIR_1>
   ```

   The check works from the shard headers and exits 1 if a rank's share would be wrong. It notes that the converter
   writes the MTP layer's small tensors into two shards; the copies are identical, and the split keeps one.
   Rank 0's folder goes on the head node, rank 1's on the other.

## Run it

Two GB10 nodes with a direct RoCE link. On each node, a container of `nvcr.io/nvidia/pytorch:26.07-py3` started with
`--gpus all --network host --ipc host --ulimit memlock=-1 --device /dev/infiniband`, the engine at `72f16e48` in
`<TENSORFOLD>`, the node's rank folder in `<RANK_DIR>` on local disk, and nothing else on the GPU.

1. Once per container, build the engine's CUDA extensions with no model loaded:
   `PYTHONPATH=<TENSORFOLD>/src python3 serve/prebuild.py`. Otherwise the server builds them after the weights are in,
   when the node no longer has the few GiB the compiler needs.
2. Before each start, free the page cache on both nodes, as root on the host: `sync; echo 3 > /proc/sys/vm/drop_caches`.
   The GPU allocates from the same memory.
3. Keep the rank folder out of the page cache while the server loads and runs:
   `serve/evict_page_cache.sh '<RANK_DIR>/*.safetensors' <STOP_FILE> &`.
4. Start both ranks (we start them within a second of each other), with `R` 0 on the head node and 1 on the other:

   ```
   TENSORFOLD_MEMORY_RESERVE_GIB=4 TF_LING_CACHE_GIB=0 TF_LING_LATENT_MB=270 TF_LING_FILLED=1 TF_GLM_UNPACK_CACHE_MB=0 \
   TF_LING_SHARED_SCRATCH=1 TF_LING_EMBED_SPLIT=1 \
   NCCL_BUFFSIZE=1048576 NCCL_MAX_NCHANNELS=4 MALLOC_ARENA_MAX=2 TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0 \
   NCCL_IB_HCA=<HCA_PORT> NCCL_IB_GID_INDEX=5 NCCL_SOCKET_IFNAME=<IFACE> PYTHONPATH=<TENSORFOLD>/src \
   python3 -m tensorfold serve <RANK_DIR> --backend cuda --tp 2 --rank R --master <HEAD_IP> --master-port 29577 \
     --host 127.0.0.1 --port 8322 --context 262144 \
     --max-tokens 4096 --answer-reserve 0.25 --answer-reserve-min 1024
   ```

   Rank 0 serves an OpenAI-compatible API at `http://127.0.0.1:8322/v1`; rank 1 follows it.
5. Beside each rank, run the watchdog, which stops the server if the node's free memory falls under 1.5 GiB:
   `serve/watchdog.sh <SERVER_PID> <STOP_FILE> <LOG> 1.5 [<CONTAINER>] &`. It never fired on this configuration.

The settings:

- `--context 262144`: the 262,144 positions of the model's config (`max_position_embeddings`; inclusionAI's model
  card names a 1M-token window, which this recipe does not run). Prompt and reply share it.
- `--max-tokens 4096`: the reply length of a request that sends no `max_tokens` (it is held to the room the window
  has left).
- `--answer-reserve 0.25 --answer-reserve-min 1024`: a reply that thinks keeps a quarter of its `max_tokens` (at least
  1,024 tokens, at most half) for the answer; see [Thinking and max_tokens](#thinking-and-max_tokens).
- `TENSORFOLD_MEMORY_RESERVE_GIB=4`: the memory the start-up check keeps free.
- `TF_LING_CACHE_GIB=0`: no memory for other conversations' prompts. The live conversation's prompt is still reused.
- `TF_LING_LATENT_MB=270`: the latent attention's working memory; at 262K it works in blocks of 8 rows, the widest
  window drafted decoding verifies (256 would cap it at 7).
- `TF_LING_SHARED_SCRATCH=1`: the decode window, the MTP layer and the prompt chunks share one set of call-local
  scratch (the latent attention's partials and the EXL3 layers' working buffers) instead of holding three; 0.66 GiB a
  node at 262K.
- `TF_LING_EMBED_SPLIT=1`: each node holds half of the embedding table and the two gather each token's row from the
  node that holds it; 0.60 GiB a node, for one small gather over the link a forward (about 0.3% of a drafted round in
  our runs).
- `TF_LING_FILLED=1`: decoding reads only the filled 512-key chunks of the cache, not the whole window.
- `TF_GLM_UNPACK_CACHE_MB=0`: no cache of unpacked weights for the prompt path, which never reuses an entry (0.375 GiB).
- `NCCL_BUFFSIZE=1048576 NCCL_MAX_NCHANNELS=4`: smaller NCCL buffers and four channels save ~1.25 GiB a rank; in our
  tests four channels were also faster on the 8-row and 2,048-row gathers.
- `MALLOC_ARENA_MAX=2`: a smaller host heap.
- `TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0`: keeps fp32 matrix products in fp32 (NVIDIA's container sets it to 1).

None of these changes the engine's output: the 40 quality prompts gave bit-identical logits with and without the
shared scratch and the split embedding, and at a 192K window the server's 18 test replies with both were token for
token the same as with v0.1.0's settings. The answer reserve changes only a reply that is still thinking at its
limit.

The draft depth adapts by default. A request with `"tf_policy": "0"` decodes serially (our serial rows used it), and
`"tf_policy": "N"` drafts N tokens a round; `--mtp-drafts N` makes that the server's default.

Hosts and addresses are placeholders; substitute your own.

## Thinking and max_tokens

Ling-3.1-flash-1019 thinks before it answers unless a request turns thinking off
(`"chat_template_kwargs": {"enable_thinking": false}`; on the Messages API it thinks only when the request
enables `thinking`), and the thinking and the answer share the request's `max_tokens`. Without a limit of its
own, a long thought can use up the whole reply and leave the answer empty.

So the server keeps part of `max_tokens` for the answer: a reply still thinking at `max_tokens` minus that reserve has
its thinking closed there (the server writes `</think>`), and the model answers in the tokens that are left. The
reserve is a quarter of `max_tokens`, at least 1,024 tokens and at most half of it:

| `max_tokens` | Thinks for at most | Kept for the answer |
|---|---|---|
| 128 | 64 | 64 |
| 512 | 256 | 256 |
| 2,048 | 1,024 | 1,024 |
| 4,096 (a request without `max_tokens`) | 3,072 | 1,024 |
| 8,192 | 6,144 | 2,048 |
| 32,768 | 24,576 | 8,192 |

If the model would end the reply right after the close, before the first word of an answer, the server starts the
answer with "Answer:" instead, once; in our checks the model always went on to answer after the close (38 of 38
replies closed after 32 or 64 thinking tokens), so this has not been needed. The model can still call a tool after the
close. A reply that closes its own thinking within the limit is not changed.

A reply whose thinking the server closed says so: `usage.completion_tokens_details` carries `"reasoning_cut": true`
beside `reasoning_tokens` (on the Messages API `usage.output_tokens_details.thinking_cut`, on the Responses API
`usage.output_tokens_details.reasoning_cut`), the server's log line for the reply carries `think_cut=<n>`, and
`finish_reason` stays what ended the reply.

To give a long thought more room, send a larger `max_tokens`. To stop thinking sooner, send `thinking_budget`
(Messages API: `thinking.budget_tokens`): a smaller budget wins, a larger one is held to the same limit. On the server,
`--answer-reserve R` sets the share (0.25, at most 0.5), `--answer-reserve-min N` the floor (1024), and
`--answer-reserve 0` turns the reserve off.

**Checked** with this engine on two GB10 nodes at the settings in [Run it](#run-it): greedy, thinking on,
`--max-tokens 4096 --answer-reserve 0.25 --answer-reserve-min 1024`. "Without" is the engine just before this change.

| Check | Without the reserve | With it |
|---|---|---|
| A 255,112-token document with two questions about it, `max_tokens` 128 | an empty answer: all 128 tokens went to thinking | thinking closed after 64 tokens; the answer names both (the reply ends at 128 tokens) |
| The same at `max_tokens` 512 and 4,096, and with no `max_tokens` | | the reply closes its own thinking (151-152 tokens) and names both |
| 20 everyday prompts at `max_tokens` 128, 400 and 700 | 18 answer; 2 think for all 700 tokens | 20 answer; 9 have their thinking closed, and the other 11 are the same, token for token |
| The same 20 at `max_tokens` 64 | 3 answer | 20 answer (19 closed) |
| A coding agent's task with tools (list, read and write files, run the tests), `max_tokens` 8,192 | | 7 turns, each answers or calls a tool; the tests pass at the end |
| One request each on the Messages API (`budget_tokens` 50 and 200, and without thinking), the Responses API and a streamed chat, `max_tokens` 256 | | each answers; usage marks the closed thinking |

Closing the thought costs a prefill of the reply so far before the answer: a reply whose thinking was closed took
0.1-0.3 s longer than a reply of the same length without the reserve after 32-64 thinking tokens, and 0.9 s longer
after 349. A reply that closes its own thinking took the same time, within 2 %.

## Limits

- **Memory margin.** The weights take 103.4 GiB of each node's 123.79 GiB, and the server's buffers, caches and graphs
  4.6 GiB more, so a node keeps only ~2-4 GiB free while serving (lowest 1.94 GiB in our checks). Nothing else can use
  the GPU, and the nodes' own memory use has to stay small. Our watchdog stops the server at 1.5 GiB.
- **Long prompts are slow.** The nine MLA layers attend over the whole prompt, so their share of the work grows with
  its length: a 255,113-token prompt took 18.8 minutes to its first token (226 tok/s). `TF_LING_PREFILL_ROWS` sets
  other prompt chunk sizes, but it changes the output bits, so this recipe keeps the default 2,048 rows.
- **One request at a time.** The engine decodes one request at a time; the others wait their turn. Serve without
  `--parallel`.
- **A repeated prompt can answer differently.** When a prompt reuses the previous one, only its new tokens are
  computed, in a chunk of fewer rows. The kernels depend on a chunk's row count, so those tokens' logits can differ
  slightly from a fresh run's, and so can the reply.
- **Quality against the original model** was not measured end to end; the numbers above compare the engine with a
  reference forward of the same 3.03 bpw weights.
- **Start-up** takes ~5 minutes, and the CUDA extensions must be built first (step 1 of Run it).
- **Tested** only with our conversion, on two GB10 nodes joined by one direct RoCE link, in
  `nvcr.io/nvidia/pytorch:26.07-py3`.

## Credits

- **inclusionAI** — Ling-3.1-flash-1019 and its weights (MIT), and its modeling code (`modeling_bailing_moe_v3.py`,
  Apache-2.0, Copyright 2025 Antgroup and The HuggingFace Inc. team).
- **ashhart** — [TensorFold](https://github.com/ashhart/TensorFold) (Apache-2.0), the engine this family plugs into.
- **turboderp** — [EXL3 / exllamav3](https://github.com/turboderp-org/exllamav3) (MIT), the weight format and the
  converter our Ling module extends.
- **flash-linear-attention** ([fla-org](https://github.com/fla-org/flash-linear-attention), MIT) — the KDA kernels
  of our reference forward.
- **jayleaton** — the CUDA server's `/tokenize` route and its token-ID prompt checks follow a patch of
  [jayleaton/glm53-tensorfold-spark](https://github.com/jayleaton/glm53-tensorfold-spark) (Apache-2.0), ported by hand.
- **Bertholomus AI** (Albert Lee, [bertholomus](https://github.com/bertholomus)) — the `bailing_v3` family, the Ling
  module for exllamav3, the conversion, the deployment and the measurements.

The engine branch's `NOTICE` and `THIRD_PARTY_NOTICES.md` list the rest of the engine's credits and licenses.

Not affiliated with or endorsed by inclusionAI, NVIDIA, the TensorFold authors or the exllamav3 authors.

## License

Recipe documentation and scripts: Apache-2.0 (`LICENSE`). The weights stay under the model's MIT license.
