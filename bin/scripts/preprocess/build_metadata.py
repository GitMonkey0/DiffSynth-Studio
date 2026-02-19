import os
import json
import random
from pathlib import Path
from tqdm import tqdm
import argparse  # NEW


def scan_first_frames(first_frame_dir):
    """扫描首帧图像目录，保持嵌套结构"""
    frames = {}
    first_frame_dir = Path(first_frame_dir)

    for frame_path in first_frame_dir.rglob('*.jpg'):
        rel_path = frame_path.relative_to(first_frame_dir)
        frame_id = str(rel_path.with_suffix(''))  # 去掉.jpg扩展名
        frames[frame_id] = str(frame_path)

    return frames


def scan_control_videos(control_dir):
    """扫描控制视频目录，保持嵌套结构（与首帧/原视频一致）"""
    videos = {}
    control_dir = Path(control_dir)

    for video_path in control_dir.rglob('*.mp4'):
        rel_path = video_path.relative_to(control_dir)
        video_id = str(rel_path.with_suffix(''))  # 去掉.mp4扩展名，如 task/0
        videos[video_id] = str(video_path)

    return videos


def scan_original_videos(original_dir):
    """扫描原始视频目录，保持嵌套结构"""
    videos = {}
    original_dir = Path(original_dir)

    for video_path in original_dir.rglob('*.mp4'):
        rel_path = video_path.relative_to(original_dir)
        video_id = str(rel_path.with_suffix(''))  # 去掉.mp4扩展名
        videos[video_id] = str(video_path)

    return videos


def generate_metadata_jsonl(first_frame_dir, control_dir, original_dir, output_dir, test_size=100):
    # 扫描文件
    first_frames = scan_first_frames(first_frame_dir)
    control_videos = scan_control_videos(control_dir)
    original_videos = scan_original_videos(original_dir)

    print(f"找到 {len(first_frames)} 个首帧图像")
    print(f"找到 {len(control_videos)} 个控制视频")
    print(f"找到 {len(original_videos)} 个原始视频")

    # 匹配数据
    matched_data = []
    unmatched_frames = []
    unmatched_videos = []
    unmatched_originals = []

    for frame_id, frame_path in first_frames.items():
        control_path = control_videos.get(frame_id)
        original_path = original_videos.get(frame_id)

        if control_path and original_path:
            data_entry = {
                "reference_image": frame_path,
                "control_video": control_path,
                "video": original_path,
                "prompt": f"",
                "video_id": frame_id
            }
            matched_data.append(data_entry)
        else:
            unmatched_frames.append(frame_id)
            if not control_path:
                unmatched_videos.append(frame_id)   # FIX: normalized_id -> frame_id
            if not original_path:
                unmatched_originals.append(frame_id)

    first_frame_ids = set(first_frames.keys())

    for video_id in control_videos:
        if video_id not in first_frame_ids:
            unmatched_videos.append(video_id)

    for video_id in original_videos:
        if video_id not in first_frame_ids:
            unmatched_originals.append(video_id)

    print(f"成功匹配 {len(matched_data)} 个数据对")
    if unmatched_frames:
        print(f"未匹配的首帧图像（{len(unmatched_frames)}个）: {unmatched_frames[:5]}...")
    if unmatched_videos:
        print(f"未匹配的控制视频（{len(unmatched_videos)}个）: {unmatched_videos[:5]}...")
    if unmatched_originals:
        print(f"未匹配的原始视频（{len(unmatched_originals)}个）: {unmatched_originals[:5]}...")

    if len(matched_data) == 0:
        print("错误：没有匹配的数据对！")
        return None, None

    # 随机分割数据
    random.shuffle(matched_data)

    actual_test_size = min(test_size, len(matched_data) // 2)
    if actual_test_size < test_size:
        print(f"警告：数据量不足，测试集大小调整为 {actual_test_size}")

    test_data = matched_data[:actual_test_size]
    train_data = matched_data[actual_test_size:]

    # 创建输出目录
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 写入训练集
    train_path = output_dir / "train_metadata.jsonl"
    with open(train_path, 'w', encoding='utf-8') as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    # 写入测试集
    test_path = output_dir / "test_metadata.jsonl"
    with open(test_path, 'w', encoding='utf-8') as f:
        for item in test_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"\n数据集已生成：")
    print(f"训练集: {train_path} ({len(train_data)} 条)")
    print(f"测试集: {test_path} ({len(test_data)} 条)")

    return str(train_path), str(test_path)


def validate_metadata(jsonl_path):
    """验证生成的JSONL文件"""
    print(f"\n验证 {jsonl_path}...")
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f, 1):
            try:
                data = json.loads(line.strip())
                required_fields = ["reference_image", "control_video", "video"]
                for field in required_fields:
                    if field not in data:
                        print(f"第{i}行缺少字段: {field}")
                        return False
                if not os.path.exists(data["reference_image"]):
                    print(f"第{i}行图像文件不存在: {data['reference_image']}")
                    return False
                if not os.path.exists(data["control_video"]):
                    print(f"第{i}行控制视频不存在: {data['control_video']}")
                    return False
                if not os.path.exists(data["video"]):
                    print(f"第{i}行目标视频不存在: {data['video']}")
                    return False
            except json.JSONDecodeError as e:
                print(f"第{i}行JSON格式错误: {e}")
                return False
    print(f"✓ {jsonl_path} 验证通过")
    return True


def parse_args():  # NEW
    p = argparse.ArgumentParser()
    p.add_argument("--reference_image_directory", required=True)
    p.add_argument("--control_directory", required=True)
    p.add_argument("--original_directory", required=True)
    p.add_argument("--output_directory", required=True)
    p.add_argument("--test_size", type=int, default=100)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()  # NEW

    train_file, test_file = generate_metadata_jsonl(
        args.reference_image_directory,
        args.control_directory,
        args.original_directory,
        args.output_directory,
        test_size=args.test_size
    )

    if train_file and test_file:
        validate_metadata(train_file)
        validate_metadata(test_file)