# MLA implementation notes

Scope: MLA path in Mirage's `mpk` branch (built for DeepSeek-V3), compared against DeepSeek-Coder-V2-Lite's config and HuggingFace implementation.

## Cache layout

One bf16 tensor per layer, one row per cached token: 512 values of compressed latent, 64 values of rope key, 576 total. Allocated in `demo.py`. Populated by `mla_kv_cache_gather_sm100.cuh`.

Coder-V2-Lite's `kv_lora_rank` and `qk_rope_head_dim` are 512 and 64, the same as V3. Row layout is identical between the two configs.

The rope portion is stored padded to 128 slots; the underlying value is 64. The padding value is hardcoded in two places: the cache tensor allocation and the gather kernel's task registration.

## Paging

Per-request index buffers: `qo_indptr`, `paged_kv_indptr`, `paged_kv_indices`, `paged_kv_last_page_len`. Page size is a compile-time constant. The gather kernel is the only component that reads the page table; it produces one flat buffer per step. No other kernel in the path reads paged indices.

Page size 128 with a non-page-aligned sequence length (1000) was tested in the Task 1 reference harness. No difference between V3 and Lite in this component.

## q_nope / q_rope handling

Post-absorption, query is a single 576-wide tensor per head. The 512/64 nope/rope split is positional; the attention kernel processes it as one dot product across 9 chunks of 64.

Same split, same chunking, same shapes for Lite.

`builder.py` computes a table of rotation angles (`rope_cos`, `rope_sin`) and attaches it as a graph input. No task in the graph reads that tensor. No MLA kernel signature takes a cos or sin argument. Rotation is not applied to query or key anywhere in this path.

## Latent-space attention

`mla_mtp_decode_sm100.cuh` has `NUM_HEADS=128` as a compile-time constant. A `local_num_heads` parameter exists in the function signature but the current task registration does not pass a value for it, so the default of 128 is what runs. The kernel's Q tensor-memory-accelerator descriptor is computed assuming a `[B*Q_LEN*H, 576]` input shape; the tensor the builder passes has an unpartitioned `[mbt, H*576]` shape.

`mla_mtp_decode_tp8_sm100.cuh` has `NUM_HEADS=16` as a compile-time constant. Contains no NVSHMEM calls, no rank checks, no cross-GPU code. Its Q descriptor computation matches the `[mbt, H*576]` shape the builder produces.

Coder-V2-Lite has 16 attention heads. The softmax scale constant embedded in the task registration is derived from YaRN's `mscale_all_dim`: 1.0 in V3's config, 0.707 in Lite's.

## Reductions

The KV sequence is split into 128-token chunks, one decode task per chunk. Each task writes a partial output and a log-sum-exp value. A separate merge task combines partial outputs using the log-sum-exp values, required because softmax normalization needs the full row.

Two merge kernels exist. `mla_mtp_decode_sm100.cuh`'s merge stores every chunk's log-sum-exp values in a fixed-size shared memory array (`MAX_SK=32`), bounding total context to 4096 tokens. `mla_mtp_decode_tp8_sm100.cuh`'s merge computes a running max and sum across chunks with no shared-memory staging and no fixed context bound.

`mla_mtp_decode_tp8_sm100.cuh`'s merge rounds the query count up to an even number internally. The attention kernel masks the resulting padding row using the real, unpadded query count. The merge kernel's bounds check and output-buffer stride use the padded count. At batch size 1, the padded query count is 2; the merge writes a row for query index 1 into an output buffer sized for one query.

## Output handling

The merge kernel's output shape is `[B*Q_LEN, heads, D_V]`, matching the output projection's input tensor shape. This match exists because V's decompression weight is folded into the output projection weight offline; there is no separate step that expands V back into its uncompressed per-head form.

Same shapes for Lite.

## Weight absorption / decompression

Two weight foldings occur during checkpoint conversion, not at inference time. `absorb_kv_into_q` in `convert.py` multiplies K's decompression weight into the query projection weight, valid because K appears inside the Q·K^T product, before the softmax. `demo.py` multiplies V's decompression weight into the output projection weight at load time, valid because V appears after the softmax and the matrix product is associative across that boundary. Neither computation references head count or hidden size.

V3's query path is `q_a_proj -> q_a_layernorm -> q_b_proj`, three steps through an intermediate `q_lora_rank`-sized tensor. Lite's config sets `q_lora_rank` to `None`; HuggingFace's implementation for that config uses a single `q_proj` with no intermediate step. `builder.py`'s `_build_mla_attention_layer` emits the three-step form unconditionally.

`_fp8_linear` in `builder.py` branches on whether a `weight_scale` tensor is present. When absent, it calls the bf16 linear task. Lite's checkpoint carries no FP8 scale tensors, so every call in this path takes that branch. The FP8 branch clamps `grid_dim` so no task processes fewer than 128 output rows. The bf16 branch does not apply this clamp; it uses `grid_for_rmsnorm_linear_layer`'s output directly, which for `kv_a_latent` (output size 512) produces 8 rows per task.

## SM100 assumptions

`tcgen05` tensor memory allocation and MMA instructions, TMA descriptors with 128-byte swizzle, a two-warp producer/consumer split for load and compute, and a shared memory budget computed for SM90 and above (roughly 201 KB after a 16 KB static allocation tied to `MAX_SK=32`). None of these values are read from model config.

Unchanged between V3 and Lite. No SM90 or SM80 code path exists for MLA in this tree.

---

## Table

| component | keep / change / new | what to do |
|---|---|---|
| KV cache layout | keep | no difference between configs |
| paging / gather | keep | rope padding value (128) is hardcoded in two places |
| q_nope / q_rope split | keep | no difference between configs |
| RoPE | new | no rotation task or kernel exists in the current path |
| attention kernel | change | `mla_mtp_decode_tp8_sm100.cuh` matches Lite's head count and Q shape; `mla_mtp_decode_sm100.cuh` does not. Softmax scale constant differs by `mscale_all_dim` |
| reduction / merge | change | `mla_mtp_decode_tp8_sm100.cuh`'s merge has no context-length bound; it has the batch-size-1 output bounds issue described above |
| output handling | keep | output shape matches the merge kernel's output shape |
| weight absorption (K, V) | keep | folding math does not reference head count or hidden size |
| query projection path | change | Lite's config has `q_lora_rank=None` (single `q_proj`); `builder.py` emits V3's three-step form unconditionally |
| bf16 linear path | change | bf16 branch of `_fp8_linear` has no 128-row task-size clamp; the FP8 branch does |
| SM100 kernel machinery | keep | no model-specific values referenced |
| model builder | new | no `deepseek_v3`-style builder exists for this model |
| weight converter | new | no `convert.py`-style script exists for this model |

keep: 6. change: 4. new: 3.
