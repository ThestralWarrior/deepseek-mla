# BF16 Tolerance - MLA Decode

`python mla_decode_reference.py --seed 1234`
transformers 5.16.1, DeepseekV2Attention eager, batch=1

```
heads=16 kv_lora_rank=512 qk_rope=64 v_head=128
attention scaling = 0.1147213868  (1/sqrt(192) would be 0.0721687836)
```

## Correctness gate (cached decode vs cacheless recompute, fp32)

```
n=16     attn 1.79e-07    layer 2.38e-07
n=128    attn 6.71e-08    layer 2.38e-07
n=1000   attn 3.73e-08    layer 2.38e-07
n=1024   attn 3.54e-08    layer 2.38e-07
```

## Per-length results (seed 1234)

```
   len tensor             max|Δ|    mean|Δ|     Δ/rms  atol@1e-2
    16 latent cache    1.654e-02  2.526e-03  1.65e-02  7.243e-03
    16 kpe cache       7.223e-03  1.442e-03  1.20e-02  3.892e-03
    16 q_nope          6.807e-03  1.260e-03  1.13e-02  3.196e-03
    16 q_pe            7.739e-03  1.331e-03  1.40e-02  3.416e-03
    16 attn out        1.809e-03  4.097e-04  1.88e-02  1.377e-03
    16 layer out       8.397e-03  1.261e-03  8.29e-03  1.284e-03
   128 latent cache    2.136e-02  2.468e-03  2.14e-02  8.402e-03
   128 kpe cache       1.177e-02  1.440e-03  2.04e-02  4.460e-03
   128 q_nope          6.419e-03  1.163e-03  1.08e-02  2.447e-03
   128 q_pe            2.429e-02  1.277e-03  3.75e-02  3.672e-03
   128 attn out        6.521e-04  1.520e-04  1.86e-02  5.413e-04
   128 layer out       7.949e-03  1.174e-03  7.99e-03  4.719e-04
  1000 latent cache    2.215e-02  2.488e-03  2.22e-02  1.015e-02
  1000 kpe cache       1.177e-02  1.462e-03  2.03e-02  5.575e-03
  1000 q_nope          3.199e-02  1.202e-03  5.20e-02  2.300e-03
  1000 q_pe            8.379e-03  1.375e-03  1.46e-02  2.975e-03
  1000 attn out        2.762e-04  5.944e-05  1.90e-02  1.964e-04
  1000 layer out       7.771e-03  1.133e-03  7.76e-03  1.123e-04
  1024 latent cache    2.215e-02  2.488e-03  2.22e-02  1.015e-02
  1024 kpe cache       1.177e-02  1.463e-03  2.03e-02  5.575e-03
  1024 q_nope          1.007e-02  1.303e-03  1.64e-02  3.264e-03
  1024 q_pe            8.240e-03  1.430e-03  1.38e-02  2.862e-03
  1024 attn out        2.735e-04  6.034e-05  1.84e-02  2.381e-04
  1024 layer out       7.632e-03  1.115e-03  7.78e-03  1.704e-04
```

## BF16 error floor (worst across all 4 lengths, rtol=1e-2)

```
latent cache    atol >= 1.015e-02
kpe cache       atol >= 5.959e-03
q_nope          atol >= 4.392e-03
q_pe            atol >= 3.680e-03
attn out        atol >= 1.377e-03
layer out       atol >= 1.284e-03
```

## Seed sweep - 8 seeds, worst-across-lengths per seed

```
seed=     0  attn=1.466e-03  layer=1.099e-03  latent=1.057e-02  kpe=6.832e-03
seed=     1  attn=1.117e-03  layer=8.036e-04  latent=9.302e-03  kpe=5.427e-03
seed=    42  attn=1.269e-03  layer=1.035e-03  latent=1.022e-02  kpe=5.482e-03
seed=   123  attn=1.571e-03  layer=1.030e-03  latent=1.084e-02  kpe=5.052e-03
seed=  1234  attn=1.377e-03  layer=1.284e-03  latent=1.015e-02  kpe=5.575e-03
seed=  4242  attn=1.664e-03  layer=1.312e-03  latent=1.168e-02  kpe=5.538e-03
seed=  7777  attn=1.495e-03  layer=1.928e-03  latent=1.058e-02  kpe=5.272e-03
seed= 99999  attn=1.330e-03  layer=1.223e-03  latent=1.152e-02  kpe=6.536e-03

              min        max        mean
attn_out      1.117e-03  1.664e-03  1.411e-03
layer_out     8.036e-04  1.928e-03  1.214e-03
latent_cache  9.302e-03  1.168e-02  1.061e-02
kpe_cache     5.052e-03  6.832e-03  5.714e-03
q_nope        3.264e-03  5.671e-03  4.502e-03
q_pe          3.645e-03  5.374e-03  4.168e-03
```

## Recommended tolerance

```python
torch.testing.assert_close(mirage_out, ref_out, rtol=1e-2, atol=<see table>)
```

| tensor | atol |
|---|---|
| latent cache | 1.3e-2 |
| kpe cache | 7.5e-3 |
| q_nope | 6e-3 |
| q_pe | 6e-3 |
| attn output | 2e-3 |
| layer output | 2.5e-3 |

(seed-sweep max, rounded up)

