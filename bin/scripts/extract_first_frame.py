import os  
from pathlib import Path  
from tqdm import tqdm  
from diffsynth.utils.data import LowMemoryVideo  
  
def extract_first_frames_with_diffsynth(video_dir, output_dir):  
    """  
    使用DiffSynth-Studio工具提取首帧  
    """  
    video_dir = Path(video_dir)  
    output_dir = Path(output_dir)  
    output_dir.mkdir(parents=True, exist_ok=True)  
      
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.webm'}  
      
    extracted_count = 0  
    failed_count = 0  
      
    # 查找所有视频文件  
    video_files = []  
    for ext in video_extensions:  
        video_files.extend(video_dir.rglob(f'*{ext}'))  
      
    print(f"找到 {len(video_files)} 个视频文件")  
      
    for video_path in tqdm(video_files, desc="提取首帧"):  
        try:  
            # 使用LowMemoryVideo读取视频  
            video = LowMemoryVideo(str(video_path))  
              
            if len(video) > 0:  
                # 获取第一帧  
                first_frame = video[0]  
                  
                # 计算输出路径  
                rel_path = video_path.relative_to(video_dir)  
                output_subdir = output_dir / rel_path.parent  
                output_subdir.mkdir(parents=True, exist_ok=True)  
                  
                output_filename = video_path.stem + '.jpg'  
                output_path = output_subdir / output_filename  
                  
                # 保存首帧  
                first_frame.save(output_path)  
                extracted_count += 1  
            else:  
                print(f"警告：视频为空 {video_path}")  
                failed_count += 1  
                  
        except Exception as e:  
            print(f"处理视频 {video_path} 时出错: {e}")  
            failed_count += 1  
      
    print(f"\n提取完成！")  
    print(f"成功提取: {extracted_count} 个首帧")  
    print(f"失败: {failed_count} 个")  
    print(f"首帧保存到: {output_dir}")  
  
if __name__ == "__main__":  
    video_directory = "/mnt/bn/aicoding-lq/luhaotian/data/test"  
    output_directory = "/mnt/bn/aicoding-lq/luhaotian/data/test_first_frame"  
      
    extract_first_frames_with_diffsynth(video_directory, output_directory)