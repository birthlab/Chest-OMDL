#!/usr/bin/env python3
"""
preprocess_ct.py - CT图像预处理独立脚本（支持断点续传和多进程）
将MHA和NII.GZ文件统一预处理并保存为H5格式
"""

# 设置线程限制 - 必须在其他导入之前
import os
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["VECLIB_MAXIMUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"

import argparse
import h5py
import numpy as np
import torch
import torch.nn.functional as F
import nibabel as nib
import SimpleITK as sitk
import cv2
import torchio as tio
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool, Manager
import warnings
import traceback

warnings.filterwarnings("ignore")

# 限制PyTorch线程数
torch.set_num_threads(2)
torch.set_num_interop_threads(1)

# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------
def strip_known_suffixes(name: str) -> str:
    """移除文件名的已知后缀"""
    n = name
    while True:
        nl = n.lower()
        matched = False
        for sfx in (".nii.gz", ".nii", ".mha", ".mhd"):
            if nl.endswith(sfx):
                n = n[: -len(sfx)]
                matched = True
                break
        if not matched:
            break
    return n

def normalize_stem(p) -> str:
    """标准化文件名（与原代码保持一致）"""
    return strip_known_suffixes(Path(p).name)

def resize_array(array, current_spacing, target_spacing):
    """重采样数组到目标体素间距"""
    original_shape = array.shape[2:]  # (D,H,W)
    scaling_factors = [current_spacing[i] / target_spacing[i] for i in range(len(original_shape))]
    new_shape = [int(original_shape[i] * scaling_factors[i]) for i in range(len(original_shape))]
    resized_array = F.interpolate(array, size=new_shape, mode='trilinear', align_corners=False).cpu().numpy()
    return resized_array

# ---------------------------------------------------------------------------
# MHA文件处理函数
# ---------------------------------------------------------------------------
def _load_mha_as_tensor(file_path):
    """使用MONAI加载MHA文件"""
    import monai
    from monai.data import ITKReader
    
    monai_loader = monai.transforms.Compose([
        monai.transforms.LoadImaged(keys=['image'], reader=ITKReader()),
        monai.transforms.EnsureChannelFirstd(keys=['image']),
        monai.transforms.Orientationd(axcodes="LPS", keys=['image']),
        monai.transforms.EnsureTyped(keys=["image"], dtype=torch.float32),
    ])
    dictionary = monai_loader({'image': file_path})
    return dictionary['image']  # (C,D,H,W)

def _get_spacing_from_itk(file_path):
    """从ITK图像获取体素间距"""
    image = sitk.ReadImage(str(file_path))
    spacing = image.GetSpacing()  # (x, y, z)
    return spacing[2], spacing[1], spacing[0]  # → return (z, y, x)

def preprocess_mha_file(file_path):
    """预处理MHA文件"""
    file_path_str = str(file_path)
    
    # 加载MHA文件
    img_data = _load_mha_as_tensor(file_path_str)  # (C,D,H,W) 即 (C,x,y,z)
    
    # 获取体素间距
    current = _get_spacing_from_itk(file_path_str)  # (z, y, x)
    target = (3.0, 1.0, 1.0)
    
    # 裁剪HU值到[-1000, 1000]
    img_data = torch.clamp(img_data, min=-1000, max=1000)
    
    # 转换为numpy并换维度
    img_np = img_data[0].cpu().numpy()     # (D,H,W) 即 (x,y,z)
    img_np = img_np.transpose(2, 0, 1)     # (W,D,H) 即 (z,x,y)
    tensor = torch.tensor(img_np).unsqueeze(0).unsqueeze(0)  # (1,1,W,D,H) 即 (1,1,z,x,y)
    
    # 重采样到目标间距
    resized_array = resize_array(tensor, current, target)    # (1,1,W',D',H') 即 (1,1,z',x',y')
    resized_array = resized_array[0][0]                      # (W',D',H') 即 (z',x',y')
    
    # 翻转y轴和x轴
    resized_array = np.flip(resized_array, axis=2)  # 在y维度上翻转
    resized_array = np.flip(resized_array, axis=1)  # 在x维度上翻转
    
    return {
        'data': resized_array.astype(np.float32),  # (z,x,y)格式
        'spacing': (np.float32(1.0), np.float32(1.0), np.float32(1.0)),
        'original_spacing': current,
    }

# ---------------------------------------------------------------------------
# NII.GZ文件处理函数
# ---------------------------------------------------------------------------
def _load_nii_as_tensor(file_path):
    """使用MONAI加载NII文件"""
    import monai
    from monai.data import ITKReader
    
    monai_loader = monai.transforms.Compose([
        monai.transforms.LoadImaged(keys=['image'], reader=ITKReader()),
        monai.transforms.EnsureChannelFirstd(keys=['image']),
        monai.transforms.Orientationd(axcodes="LPS", keys=['image']),
        monai.transforms.EnsureTyped(keys=["image"], dtype=torch.float32),
    ])
    dictionary = monai_loader({'image': file_path})
    return dictionary['image']

def preprocess_nii_file(file_path):
    """预处理NII.GZ文件 - 使用与MHA相同的流程"""
    file_path_str = str(file_path)
    
    # 加载NII文件
    img_data = _load_nii_as_tensor(file_path_str)
    
    # 获取体素间距
    current = _get_spacing_from_itk(file_path_str)
    target = (3.0, 1.0, 1.0)
    
    # 裁剪HU值到[-1000, 1000]
    img_data = torch.clamp(img_data, min=-1000, max=1000)
    
    # 转换为numpy并换维度
    img_np = img_data[0].cpu().numpy()
    img_np = img_np.transpose(2, 0, 1)
    tensor = torch.tensor(img_np).unsqueeze(0).unsqueeze(0)
    
    # 重采样到目标间距
    resized_array = resize_array(tensor, current, target)
    resized_array = resized_array[0][0]
    
    # 翻转y轴和x轴
    resized_array = np.flip(resized_array, axis=2)
    # resized_array = np.flip(resized_array, axis=1)
    
    return {
        'data': resized_array.astype(np.float32),
        'spacing': (np.float32(1.0), np.float32(1.0), np.float32(1.0)),
        'original_spacing': current,
    }

# ---------------------------------------------------------------------------
# 图像增强和归一化
# ---------------------------------------------------------------------------
def adaptive_windowing(image, window_min=-2000, window_max=1000):
    """应用自适应窗口化处理"""
    img_filtered = np.array(image, dtype=np.float32)
    img_filtered[img_filtered < img_filtered.min() + 100] = np.nan
    
    min_display = np.nanpercentile(img_filtered, 0.5)
    max_display = np.nanpercentile(img_filtered, 99.5)
    
    img_windowed = np.clip(image, min_display, max_display)
    img_windowed = ((img_windowed - min_display) / (max_display - min_display) * 255).astype(np.uint8)
    
    return img_windowed

def apply_clahe(img, clip_limit=2.0, tile_grid_size=(8, 8)):
    """应用CLAHE直方图均衡化"""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    clahe_slices = []
    for i in range(img.shape[0]):
        slice_img = img[i, :, :]
        clahe_img = clahe.apply(slice_img)
        clahe_slices.append(clahe_img)
    return np.stack(clahe_slices, axis=0)

def normalize_data(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    """归一化处理"""
    image_windowed = adaptive_windowing(image)
    image_clahe = apply_clahe(image_windowed, clip_limit, tile_grid_size)
    image_normalized = image_clahe.astype(np.float32) / 255.0
    image_normalized = np.expand_dims(image_normalized, axis=0)
    return image_normalized

def resize_data(image, target_shape=(64, 128, 128)):
    """调整图像大小"""
    resize_transform = tio.Resize(target_shape)
    return resize_transform(image)

def resample_data(image, target_spacing=(1, 1, 1)):
    """重采样图像"""
    resample_transform = tio.Resample(target_spacing)
    return resample_transform(image)

# ---------------------------------------------------------------------------
# 单文件处理函数（用于多进程）
# ---------------------------------------------------------------------------
def process_single_file(args):
    """
    处理单个文件的函数（支持多进程）
    """
    file_path, output_path, visualize_dir, should_visualize, process_lock = args
    
    try:
        file_path_str = str(file_path)
        file_key = normalize_stem(file_path)
        
        # 第一阶段：根据文件类型选择处理方法
        if file_path_str.lower().endswith('.mha'):
            processed_data = preprocess_mha_file(file_path)
        elif file_path_str.lower().endswith('.nii.gz') or file_path_str.lower().endswith('.nii'):
            processed_data = preprocess_nii_file(file_path)
        else:
            return f"✗ 不支持的文件格式: {file_path.name}", None
        
        ct_img = processed_data['data']
        spacing = processed_data['spacing']
        
        # 第二阶段：归一化处理
        ct_normalized = normalize_data(ct_img)
        
        # 第三阶段：创建torchio对象
        ct_img_tensor = torch.from_numpy(ct_normalized)
        ct_subject = tio.Subject(
            image=tio.ScalarImage(
                tensor=ct_img_tensor,
                spacing=spacing
            )
        )
        
        # 第四阶段：重采样和调整大小
        ct_resampled_subject = resample_data(ct_subject.image)
        ct_resized_subject = resize_data(ct_resampled_subject, target_shape=(64, 128, 128))
        
        ct_tensor = ct_resized_subject.data.numpy()
        
        # 使用进程锁保护HDF5文件写入
        with process_lock:
            with h5py.File(output_path, 'a') as hf:
                if file_key not in hf:
                    # 创建数据集（使用"ct"作为数据集名称，与原代码保持一致）
                    grp = hf.create_group(file_key)
                    grp.create_dataset("ct", data=ct_tensor, compression="gzip", dtype="float32")
                    grp.create_dataset("original_spacing", data=processed_data['original_spacing'])
                    grp.create_dataset("final_spacing", data=spacing)
                    grp.attrs['filename'] = file_path.name
        
        # 可视化（如果需要）
        vis_path = None
        if should_visualize and visualize_dir:
            vis_path = Path(visualize_dir) / f"{file_key}_preprocessed.nii.gz"
            save_as_nifti(ct_tensor, vis_path)
        
        return f"✓ {file_path.name}", file_key
        
    except Exception as e:
        error_msg = f"✗ 处理 {file_path.name} 时出错: {str(e)}"
        # traceback.print_exc()
        return error_msg, None

def save_as_nifti(data, output_path):
    """保存为NIfTI文件用于可视化"""
    affine = np.eye(4)
    
    # 如果数据有通道维度，移除它
    if data.ndim == 4 and data.shape[0] == 1:
        data = data[0]
    
    # 转置为NIfTI标准格式 (x, y, z)
    data_transposed = np.transpose(data, (1, 2, 0))
    
    nii_img = nib.Nifti1Image(data_transposed, affine)
    nib.save(nii_img, output_path)

# ---------------------------------------------------------------------------
# 主函数
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="CT图像预处理脚本（支持断点续传）")
    parser.add_argument("--input", type=str, required=True,
                       help="输入CT文件目录")
    parser.add_argument("--output", type=str, required=True,
                       help="输出H5文件路径")
    parser.add_argument("--visualize_dir", type=str, default=None,
                       help="前N个样本的可视化输出目录（可选）")
    parser.add_argument("--num_visualize", type=int, default=10,
                       help="需要可视化的样本数量（默认10）")
    parser.add_argument("--num_processes", type=int, default=4,
                       help="并行处理的进程数（默认4）")
    parser.add_argument("--force_reprocess", action="store_true",
                       help="强制重新处理所有文件（忽略已处理的文件）")
    
    args = parser.parse_args()
    
    # 获取所有输入文件
    input_path = Path(args.input)
    all_files = sorted([p for p in input_path.iterdir()
                       if p.suffix.lower() in {".mha", ".nii", ".gz"}
                       or (p.suffix.lower() == ".gz" and p.with_suffix("").suffix.lower() == ".nii")])
    
    if not all_files:
        raise SystemExit(f"✗ 在 {args.input} 中未找到CT体积文件")
    
    # 读取已处理的文件（断点续传）
    processed_files = set()
    output_path = Path(args.output)
    
    if not args.force_reprocess and output_path.exists():
        print("检查已处理的文件...")
        with h5py.File(output_path, 'r') as hf:
            processed_files = set(hf.keys())
        print(f"✓ 找到 {len(processed_files)} 个已处理的文件")
    
    # 确定需要处理的文件
    all_file_keys = {normalize_stem(f) for f in all_files}
    remaining_file_keys = all_file_keys - processed_files
    
    # 过滤出需要处理的文件路径
    files_to_process = [f for f in all_files if normalize_stem(f) in remaining_file_keys]
    
    print(f"\n{'='*70}")
    print(f"文件统计:")
    print(f"  - 总文件数: {len(all_files)}")
    print(f"  - 已处理: {len(processed_files)}")
    print(f"  - 待处理: {len(files_to_process)}")
    print(f"  - 并行进程数: {args.num_processes}")
    print(f"{'='*70}\n")
    
    if not files_to_process:
        print("✓ 所有文件都已处理完成！")
        return
    
    # 创建输出目录
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 确保H5文件存在
    if not output_path.exists():
        with h5py.File(output_path, 'w') as hf:
            pass  # 创建空文件
    
    # 创建可视化目录（如果需要）
    if args.visualize_dir:
        vis_dir = Path(args.visualize_dir)
        vis_dir.mkdir(parents=True, exist_ok=True)
        print(f"可视化文件将保存到: {vis_dir}\n")
    
    # 创建进程管理器和锁
    manager = Manager()
    process_lock = manager.Lock()
    
    # 准备进程池参数
    args_list = []
    for idx, file_path in enumerate(files_to_process):
        should_visualize = args.visualize_dir and (idx < args.num_visualize)
        args_list.append((
            file_path,
            str(output_path),
            args.visualize_dir,
            should_visualize,
            process_lock
        ))
    
    # 使用进程池处理文件
    successful = 0
    failed = 0
    
    print("开始预处理...\n")
    
    with Pool(processes=args.num_processes) as pool:
        for result, file_key in tqdm(
            pool.imap_unordered(process_single_file, args_list),
            total=len(args_list),
            desc="处理进度"
        ):
            if file_key:
                successful += 1
            else:
                failed += 1
            
            # 只在出错时打印详细信息
            if not file_key:
                tqdm.write(result)
    
    # 打印最终统计
    print(f"\n{'='*70}")
    print(f"预处理完成!")
    print(f"  - 成功: {successful} 个文件")
    print(f"  - 失败: {failed} 个文件")
    print(f"  - H5文件: {output_path}")
    if args.visualize_dir:
        print(f"  - 可视化文件: {args.visualize_dir}")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()