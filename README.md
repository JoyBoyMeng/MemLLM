# MemLLM

## Environment Setup

```bash
cd MemLLM

conda create -n memllm python=3.10 -y
conda activate memllm

pip install torch
pip install numpy pandas scikit-learn transformers accelerate sentencepiece
```

## Run

```bash
python train_link_prediction.py \
  --data yelpv2 \
  --llm_path /path/to/Llama-3-8B \
  --gpu 0 \
  --bs 16 \
  --mem_dim 128 \
  --n_degree 20 \
  --lr 5e-4 \
  --n_epoch 50 \
  --patience 10
```

Replace `--llm_path` with your local model path. Prepare the processed dataset before running.
