export http_proxy=http://sys-proxy-rd-relay.byted.org:8118 https_proxy=http://sys-proxy-rd-relay.byted.org:8118 no_proxy=.byted.org,.bytedance.net,.byteintl.net

torchrun --nproc_per_node=8 /mnt/bn/aicoding-lq/luhaotian/projects/DiffSynth-Studio/bin/scripts/infer/infer_control_hl.py \
        --jsonl /mnt/bn/aicoding-lq/luhaotian/projects/DiffSynth-Studio/bin/data/processed/metadata/test_metadata.jsonl \
        --out_dir /mnt/bn/aicoding-lq/luhaotian/projects/DiffSynth-Studio/bin/experiments/runs/videos_hl \
        --num_inference_steps 50 \
        --lora_path ./models/train/Wan2.1-Fun-V1.1-14B-Control_lora_hl/epoch-1.safetensors \
        --lora_alpha 1.0
