# Ling-3.1-flash on 2× DGX Spark (GB10) — our TensorFold TP2 engine

**Bertholomus AI recipe: inclusionAI's Ling-3.1-flash served tensor-parallel over two NVIDIA DGX Spark / GB10 nodes by
the model family we wrote for [TensorFold](https://github.com/ashhart/TensorFold) (by ashhart), with a 196,608-token
window and decoding drafted by the model's own MTP layer.**

> **Weights:** our EXL3 conversion (3.03 bits per weight) of
> [inclusionAI/Ling-3.1-flash](https://huggingface.co/inclusionAI/Ling-3.1-flash) (MIT) at revision `7b864daa`, made
> with the recipe in [`conversion/`](conversion/), is on Hugging Face as
> [bertholomus/Ling-3.1-flash-EXL3-3.03bpw](https://huggingface.co/bertholomus/Ling-3.1-flash-EXL3-3.03bpw) (220.6 GB).
> [Get the weights](#get-the-weights) downloads it; [Make the weights](#make-the-weights) makes it from the original.

**Engine:** commit `ef914199` on
[bertholomus/TensorFold, branch `ling-3.1-tp2`](https://github.com/bertholomus/TensorFold/tree/ling-3.1-tp2):
Bertholomus AI's `bailing_v3` family on upstream TensorFold v0.6.6's Python CUDA engine, on top of our GLM-5.3 engine
line (`glm-dsa-tp4`). It holds the code we serve and measured; only our internal notes, paths and one internal script
are taken out. The model math follows inclusionAI's modeling code for Ling-3.1-flash (`modeling_bailing_moe_v3.py`,
Apache-2.0).

Ling-3.1-flash (`BailingMoeV3ForCausalLM`) has 72 layers: 63 with KDA linear attention and 9 with gated MLA (layers 7,
15, …, 71); 4 dense MLP layers, then 68 mixture-of-experts layers with 512 routed experts (8 active) and a shared
expert; and one MTP layer. Its model card gives 560B total and 25B active parameters.

## v0.1.0 (2026-10-25, first release)

Engine `ef914199` with `--context 196608`: a 192K window, MTP drafting on (the default), one request at a time.
Measured on two GB10 nodes (TP2) joined by a direct RoCE link, at the settings in [Run it](#run-it), with our own
client: greedy (temperature 0), thinking on, one run per row unless noted.

Drafting does not change the reply: drafted and serial decoding give the same tokens on fresh prompts (6 of 6 test
prompts, and 2 of 2 replies that decode across the cache's 512-key chunk boundaries).

**Decode** (tok/s after the first token):

| Prompt tokens | Reply tokens | Drafted | Drafts accepted | Serial |
|---|---|---|---|---|
| 30-33 (4 prompts) | 35-128 | 40.0-47.7 | 69-82% | 23.6-24.1 |
| 1,012 | 167 | 47.9 | 114 / 138 | 23.9 |
| 2,222 | 60 | 40.9 | 36 / 51 | 23.8 |
| 6,623 | 60 | 39.8 | 36 / 54 | 23.9 |
| 8,121 | 640 | 47.3 | 441 / 529 | 23.7 |
| 31,361 | 69 | 40.3 | 42 / 55 | 22.5 ¹ |
| 121,778 | 61 | 32.2 | 35 / 51 | |
| 191,078 | 101 | 29.8 | 60 / 89 | |

¹ One 60-token reply that reused the drafted run's prompt.

How fast drafted decoding runs depends on how many of the MTP layer's drafts the model accepts, which depends on the
text. Sampled requests are drafted too (the server's defaults are temperature 1.0, top-k 20, top-p 0.95).

**Prompt** (time to the first token):

| Prompt tokens | Time | tok/s |
|---|---|---|
| 1,012 | 1.66-1.70 s | 595-610 |
| 2,222 | 3.48-3.50 s (3.89 s the first time) | 635-639 |
| 6,623 | 9.61-9.67 s | 685-689 |
| 8,121 | 11.55-11.58 s | 701-703 |
| 31,361 | 53.5 s | 587 |
| 121,778 | 339.3 s (5.7 min) | 359 |
| 191,078 | 685.6 s (11.4 min) | 279 |

A prompt that extends the previous request's prompt reuses it: the 6,623-token prompt sent again started in 0.07 s.

**Memory** (each node has 123.79 GiB, shared by its CPU and GPU):

| | Rank 0 | Rank 1 |
|---|---|---|
| GPU memory, weights loaded (PyTorch allocated) | 104.01 GiB | 104.01 GiB |
| GPU memory, ready to serve (buffers, caches, graphs) | 108.36 GiB | 108.36 GiB |
| Free (MemAvailable) after the weights load | 7.73 GiB | 7.82 GiB |
| Free when ready to serve | 3.78 GiB | 2.90 GiB |
| Lowest free while serving (our tests to 191K, then 2.5 h of use) | 2.08 GiB | 1.84 GiB |

Start to ready: 235 s, with the weights read from each node's local disk.

**Quality: the engine's arithmetic.** We compared the engine's next-token logits at every position of 40 prompts
(passages and chat turns, 66,103 positions) with a reference forward of the same EXL3 weights: our PyTorch
implementation of the model, written from inclusionAI's modeling code, with the weights decoded by exllamav3 and the
KDA layers run by flash-linear-attention's kernels. The reference ran once in fp32 throughout and once with bf16
activations:

| | Top-1 agreement | Mean KL (reference ‖ tested) | Max KL |
|---|---|---|---|
| Engine vs the fp32 reference | 0.9723 | 0.0140 | 17.70 |
| bf16 reference vs the fp32 reference | 0.9728 | 0.0123 | 16.06 |
| Engine vs the bf16 reference | 0.9724 | 0.0133 | 12.16 |

So the engine is about as close to exact fp32 arithmetic as the same math with bf16 activations: its KL is 1.14 times
as large, and its top-1 agreement is 0.0005 lower. These numbers measure arithmetic on the same quantized weights. How
far the 3.03 bpw conversion is from the original FP8 model was not measured end to end.

## Get the weights

Our conversion on Hugging Face is the folder our served rank folders were split from: steps 1 to 4 below are done,
and its `README.md` is our model card. Download it and the engine, then check and split the weights as in step 5:

```
hf download bertholomus/Ling-3.1-flash-EXL3-3.03bpw --local-dir <EXL3_DIR>
git clone -b ling-3.1-tp2 https://github.com/bertholomus/TensorFold <TENSORFOLD>
```

## Make the weights

1. Download [inclusionAI/Ling-3.1-flash](https://huggingface.co/inclusionAI/Ling-3.1-flash) at revision
   `7b864daa2cedb1404e12bc52899def2d151a9719` (FP8, 535 GiB):

   ```
   hf download inclusionAI/Ling-3.1-flash --revision 7b864daa2cedb1404e12bc52899def2d151a9719 --local-dir <SRC>
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
   the Ling-3.1 module, the code our conversion ran), in `nvcr.io/nvidia/pytorch:26.07-py3`, with this repository
   checked out at `<THIS_REPO>`:

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
   GB10 the conversion took 29 h 14 min in one pass and wrote 220.6 GB in 35 shards.

4. Put back the tensors the converter leaves unquantized, exactly as the BF16 copy has them:

   ```
   python3 <TENSORFOLD>/tools/ling_restore_passthrough.py <BF16_DIR> <EXL3_DIR>
   ```

   The converter stores those tensors (norms, the embedding, the router, `kv_b_proj`, the KDA's short convolutions,
   …) in its own dtype, some of them as F16. The restore writes back the BF16 copy's bytes and dtype, the three
   short convolutions as the one fused tensor the converter writes, and exits 0 only if every restored tensor reads
   back equal. On our conversion it restored 646 tensors and found 494 already identical.

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
`--gpus all --network host --ipc host --ulimit memlock=-1 --device /dev/infiniband`, the engine at `ef914199` in
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
   TENSORFOLD_MEMORY_RESERVE_GIB=4 TF_LING_CACHE_GIB=0 TF_LING_LATENT_MB=256 TF_LING_FILLED=1 TF_GLM_UNPACK_CACHE_MB=0 \
   NCCL_BUFFSIZE=1048576 NCCL_MAX_NCHANNELS=4 MALLOC_ARENA_MAX=2 TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0 \
   NCCL_IB_HCA=<HCA_PORT> NCCL_IB_GID_INDEX=5 NCCL_SOCKET_IFNAME=<IFACE> PYTHONPATH=<TENSORFOLD>/src \
   python3 -m tensorfold serve <RANK_DIR> --backend cuda --tp 2 --rank R --master <HEAD_IP> --master-port 29577 \
     --host 127.0.0.1 --port 8322 --context 196608
   ```

   Rank 0 serves an OpenAI-compatible API at `http://127.0.0.1:8322/v1`; rank 1 follows it.
5. Beside each rank, run the watchdog, which stops the server if the node's free memory falls under 1.5 GiB:
   `serve/watchdog.sh <SERVER_PID> <STOP_FILE> <LOG> 1.5 [<CONTAINER>] &`. It never fired on this configuration.

The settings:

- `--context 196608`: the 192K window. Prompt and reply share it.
- `TENSORFOLD_MEMORY_RESERVE_GIB=4`: the memory the start-up check keeps free.
- `TF_LING_CACHE_GIB=0`: no memory for other conversations' prompts. The live conversation's prompt is still reused.
- `TF_LING_LATENT_MB=256`: the latent attention's working memory; at 192K it works in blocks of 10 rows.
- `TF_LING_FILLED=1`: decoding reads only the filled 512-key chunks of the cache, not the whole window.
- `TF_GLM_UNPACK_CACHE_MB=0`: no cache of unpacked weights for the prompt path, which never reuses an entry (0.375 GiB).
- `NCCL_BUFFSIZE=1048576 NCCL_MAX_NCHANNELS=4`: smaller NCCL buffers and four channels save ~1.25 GiB a rank; in our
  tests four channels were also faster on the 8-row and 2,048-row gathers.
- `MALLOC_ARENA_MAX=2`: a smaller host heap.
- `TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0`: keeps fp32 matrix products in fp32 (NVIDIA's container sets it to 1).

None of these changes the engine's output: a run of the 40 quality prompts with these settings (and latent blocks of
9 rows, finer than 192K's 10) gave logits bit-identical to the run the quality numbers come from.

The draft depth adapts by default. A request with `"tf_policy": "0"` decodes serially (our serial rows used it), and
`"tf_policy": "N"` drafts N tokens a round; `--mtp-drafts N` makes that the server's default.

Hosts and addresses are placeholders; substitute your own.

## Limits

- **Memory margin.** The weights take 104 GiB of each node's 123.79 GiB, and the server's buffers, caches and graphs
  4.4 GiB more, so a node keeps only ~2 GiB free while serving (lowest 1.84 GiB). Nothing else can use the GPU, and
  the nodes' own memory use has to stay small. Our watchdog stops the server at 1.5 GiB.
- **192K is the ceiling with drafting.** The model's config allows 262,144 positions. A 262,144-token window needs
  about 0.8 GiB more per node than 192K, which would leave rank 1 under the watchdog's 1.5 GiB. Without the MTP layer
  (`TF_LING_MTP=0`) 262,144 should fit by our memory estimate (we have not run it), but decoding then runs at serial
  speed (22-24 tok/s instead of 30-48).
- **Long prompts are slow.** Prompt speed peaks at ~700 tok/s around 8K tokens and falls with length: 587 tok/s at
  31K, 359 at 122K and 279 at 191K, where the first token takes 11.4 minutes. The nine MLA layers attend over the
  whole prompt, so their share of the work grows with its length. `TF_LING_PREFILL_ROWS` sets other prompt chunk
  sizes, but it changes the output bits, so this recipe keeps the default 2,048 rows.
- **One request at a time.** The engine decodes one request at a time; the others wait their turn. Serve without
  `--parallel`.
- **A repeated prompt can answer differently.** When a prompt reuses the previous one, only its new tokens are
  computed, in a chunk of fewer rows. The kernels depend on a chunk's row count, so those tokens' logits can differ
  slightly from a fresh run's, and so can the reply.
- **Quality against the original model** was not measured end to end; the numbers above compare the engine with a
  reference forward of the same 3.03 bpw weights.
- **Start-up** takes ~4 minutes, and the CUDA extensions must be built first (step 1 of Run it).
- **Tested** only with our conversion, on two GB10 nodes joined by one direct RoCE link, in
  `nvcr.io/nvidia/pytorch:26.07-py3`.

## Credits

- **inclusionAI** — Ling-3.1-flash and its weights (MIT), and its modeling code (`modeling_bailing_moe_v3.py`,
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
