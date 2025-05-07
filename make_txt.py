import os

def build_video_list(root_dir, output_path='train_list.txt', video_exts={'.mp4', '.avi', '.mov'}):
    """
    根据目录结构构建 train_list.txt 或 val_list.txt。
    每一行格式为：相对路径 label_index

    Args:
        root_dir (str): 根目录，子文件夹为类别名，每类下有视频文件。
        output_path (str): 输出列表文件路径。
        video_exts (set): 允许的视频扩展名集合。
    """
    class_names = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
    class_to_idx = {cls_name: idx for idx, cls_name in enumerate(class_names)}

    with open(output_path, 'w') as f:
        for cls_name in class_names:
            cls_path = os.path.join(root_dir, cls_name)
            for video_name in os.listdir(cls_path):
                if os.path.splitext(video_name)[1].lower() in video_exts:
                    rel_path = os.path.join(cls_name, video_name)
                    label = class_to_idx[cls_name]
                    f.write(f"{rel_path} {label}\n")

    print(f"List saved to {output_path} with {len(class_to_idx)} classes.")

# 示例用法
build_video_list(root_dir='/home/nus-zwb/reuse/video/tiny-Kinetics-400', output_path='train_list.txt')
