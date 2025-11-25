import os

DATA_DIR = "fairseq_data" 
W2V_PATH = "models/wav2vec/wav2vec_small.pt"
SAVE_DIR = "models/wav2vec/checkpoints"

os.makedirs(SAVE_DIR, exist_ok=True)

cmd = f"""
fairseq-train {DATA_DIR} \
    --task text_classification \
    --arch wav2vec_classification \
    --w2v-path {W2V_PATH} \
    --num-classes 10 \
    --labels ltr \
    --fp16 \
    --max-update 20000 \
    --lr 1e-5 \
    --warmup-updates 500 \
    --max-sample-size 250000 \
    --max-tokens 140000 \
    --criterion binary_cross_entropy \
    --best-checkpoint-metric accuracy \
    --batch-size 4 \
    --log-format simple \
    --log-interval 20 \
    --save-dir {SAVE_DIR}
"""

os.system(cmd)
