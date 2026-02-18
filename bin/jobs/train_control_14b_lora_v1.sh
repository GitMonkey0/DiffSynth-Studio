export http_proxy=http://sys-proxy-rd-relay.byted.org:8118 https_proxy=http://sys-proxy-rd-relay.byted.org:8118 no_proxy=.byted.org,.bytedance.net,.byteintl.net

accelerate launch --config_file examples/wanvideo/model_training/full/accelerate_config_14B.yaml \
  examples/wanvideo/model_training/train.py \
  --dataset_base_path "" \
  --dataset_metadata_path /mnt/bn/aicoding-lq/luhaotian/projects/DiffSynth-Studio/extra/data/metadata/fun_control/baseline/train_metadata.jsonl \
  --data_file_keys "video,control_video,reference_image" \
  --height 480 \
  --width 832 \
  --dataset_repeat 1 \
  --model_paths '[   
    "/mnt/bn/aicoding-lq/luhaotian/ckpt/Wan2.1-Fun-V1.1-14B-Control/diffusion_pytorch_model.safetensors",
    "/mnt/bn/aicoding-lq/luhaotian/ckpt/Wan2.1-Fun-V1.1-14B-Control/models_t5_umt5-xxl-enc-bf16.pth",  
    "/mnt/bn/aicoding-lq/luhaotian/ckpt/Wan2.1-Fun-V1.1-14B-Control/Wan2.1_VAE.pth",  
    "/mnt/bn/aicoding-lq/luhaotian/ckpt/Wan2.1-Fun-V1.1-14B-Control/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"  
  ]' \
  --learning_rate 1e-4 \
  --num_epochs 1 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/Wan2.1-Fun-V1.1-14B-Control_lora" \
  --lora_base_model "dit" \
  --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
  --lora_rank 32 \
  --extra_inputs "control_video,reference_image"