export http_proxy=http://sys-proxy-rd-relay.byted.org:8118 https_proxy=http://sys-proxy-rd-relay.byted.org:8118 no_proxy=.byted.org,.bytedance.net,.byteintl.net

torchrun --nproc_per_node=8 /mnt/bn/aicoding-lq/luhaotian/projects/DiffSynth-Studio/bin/scripts/baseline/infer.py \
        --jsonl /mnt/bn/aicoding-lq/luhaotian/projects/DiffSynth-Studio/bin/data/processed/metadata/test_metadata.jsonl \
        --out_dir /mnt/bn/aicoding-lq/luhaotian/projects/DiffSynth-Studio/bin/experiments/baseline/runs/videos
