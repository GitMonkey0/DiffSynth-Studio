export http_proxy=http://sys-proxy-rd-relay.byted.org:8118 https_proxy=http://sys-proxy-rd-relay.byted.org:8118 no_proxy=.byted.org,.bytedance.net,.byteintl.net

torchrun --nproc_per_node=8 /mnt/bn/aicoding-lq/luhaotian/projects/DiffSynth-Studio/extra/infer/control_14b_lora.py \
        --jsonl /mnt/bn/aicoding-lq/luhaotian/projects/DiffSynth-Studio/extra/data/egodex/metadata/fun_control/baseline/test_metadata.jsonl \
        --out_dir /mnt/bn/aicoding-lq/luhaotian/projects/DiffSynth-Studio/extra/infer/outputs \
