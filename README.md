# MLA Forward Pass

Standalone PyTorch reference implementation of DeepSeek-Coder-V2-Lite's Multi-head Latent Attention (MLA), verified against the Hugging Face reference implementation.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run
```
python mla_forward_pass.py
```

## Expected Output
```
prefill (6 tok):  max diff 2.38e-07
decode (tok 7):   max diff 1.49e-07
ok
```
