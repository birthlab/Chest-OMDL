#!/usr/bin/env python3
"""
flip_ct_h5.py - 对H5文件中的所有CT数据进行flip操作（支持三视图预览）
"""

import argparse
import h5py
import numpy as np
from pathlib import Path
from tqdm import tqdm
import shutil
import matplotlib.pyplot as plt

def visualize_flip_preview(input_path, flip_axis, num_samples=10, output_dir='flip_preview'):
    """
    预览flip操作效果，生成对比图
    
    参数:
        input_path: 输入H5文件路径
        flip_axis: 要翻转的维度
        num_samples: 要可视化的样本数量
        output_dir: 可视化输出目录
    """
    
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"生成Flip预览对比图（三视图）")
    print(f"  输入文件: {input_path}")
    print(f"  Flip维度: axis={flip_axis}")
    print(f"  预览样本数: {num_samples}")
    print(f"  输出目录: {output_dir}")
    print(f"{'='*70}\n")
    
    with h5py.File(input_path, 'r') as hf:
        file_keys = list(hf.keys())[:num_samples]
        
        print(f"找到 {len(hf.keys())} 个文件，预览前 {len(file_keys)} 个\n")
        
        for idx, file_key in enumerate(tqdm(file_keys, desc="生成预览")):
            grp = hf[file_key]
            
            if 'ct' not in grp:
                print(f"⚠ 跳过 {file_key}: 没有'ct'数据集")
                continue
            
            # 读取原始数据
            ct_original = grp['ct'][:]
            
            # 生成翻转后的数据（仅用于预览）
            ct_flipped = np.flip(ct_original, axis=flip_axis)
            
            # 打印第一个文件的信息
            if idx == 0:
                print(f"\n数据形状: {ct_original.shape}")
                print(f"数据类型: {ct_original.dtype}")
                print(f"数值范围: [{ct_original.min():.2f}, {ct_original.max():.2f}]")
                print(f"翻转维度: axis={flip_axis} (大小={ct_original.shape[flip_axis]})\n")
            
            # 生成对比图
            fig = _create_comparison_figure(ct_original, ct_flipped, flip_axis, file_key)
            
            # 保存图像
            output_path = output_dir / f"{idx+1:02d}_{file_key}_flip_preview.png"
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
    
    print(f"\n{'='*70}")
    print(f"✓ 预览图生成完成!")
    print(f"  输出目录: {output_dir}")
    print(f"  生成图像: {len(file_keys)} 张")
    print(f"\n请查看预览图，如果满意请运行实际flip操作:")
    print(f"  python flip_ct_h5.py --input {input_path} --output <output.h5> --axis {flip_axis}")
    print(f"{'='*70}")

def _create_comparison_figure(ct_original, ct_flipped, flip_axis, file_key):
    """创建翻转前后的三视图对比"""
    
    # 提取三个维度的中心切片
    slices_orig = _extract_orthogonal_slices(ct_original)
    slices_flip = _extract_orthogonal_slices(ct_flipped)
    
    # 创建图形 (2行3列)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 标题
    axis_names = {0: 'Z-axis (Axial)', 1: 'X-axis (Sagittal)', 2: 'Y-axis (Coronal)'}
    flip_axis_name = axis_names.get(flip_axis, f'Axis-{flip_axis}')
    
    fig.suptitle(
        f'Flip Preview: {file_key}\n'
        f'Shape: {ct_original.shape} | Flip Axis: {flip_axis} ({flip_axis_name})',
        fontsize=14, fontweight='bold'
    )
    
    # 第一行：原始数据的三视图
    view_names = ['Axial (Z-axis)', 'Sagittal (X-axis)', 'Coronal (Y-axis)']
    view_keys = ['axial', 'sagittal', 'coronal']
    
    for i, (view_name, view_key) in enumerate(zip(view_names, view_keys)):
        ax = axes[0, i]
        img = slices_orig[view_key]
        
        # 显示图像
        im = ax.imshow(img, cmap='gray', vmin=0, vmax=1)
        ax.set_title(f'Original - {view_name}', fontsize=11, fontweight='bold')
        ax.axis('off')
        
        # 添加维度标注
        h, w = img.shape
        ax.text(5, 20, f'{h}×{w}', color='yellow', fontsize=9, 
                bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))
    
    # 第二行：翻转后的三视图
    for i, (view_name, view_key) in enumerate(zip(view_names, view_keys)):
        ax = axes[1, i]
        img = slices_flip[view_key]
        
        # 显示图像
        im = ax.imshow(img, cmap='gray', vmin=0, vmax=1)
        ax.set_title(f'Flipped - {view_name}', fontsize=11, fontweight='bold')
        ax.axis('off')
        
        # 添加维度标注
        h, w = img.shape
        ax.text(5, 20, f'{h}×{w}', color='yellow', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))
        
        # 计算并显示差异信息
        diff = np.abs(slices_orig[view_key] - img)
        mean_diff = np.mean(diff)
        max_diff = np.max(diff)
        
        # 在图像底部添加差异信息
        diff_text = f'Diff: mean={mean_diff:.4f}, max={max_diff:.4f}'
        ax.text(w/2, h-10, diff_text, color='cyan', fontsize=8, ha='center',
                bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
    
    plt.tight_layout()
    return fig

def _extract_orthogonal_slices(ct_data):
    """
    提取CT数据的三个正交切片（轴向、矢状、冠状）
    
    返回字典包含:
        - 'axial': 轴向切片 (xy平面, 沿z轴的中心切片)
        - 'sagittal': 矢状切片 (yz平面, 沿x轴的中心切片)
        - 'coronal': 冠状切片 (xz平面, 沿y轴的中心切片)
    """
    
    # 处理不同维度的数据
    if ct_data.ndim == 4:  # (C, D, H, W) 或 (1, D, H, W)
        data = ct_data[0]  # 移除通道维度 -> (D, H, W)
    elif ct_data.ndim == 3:  # (D, H, W)
        data = ct_data
    else:
        raise ValueError(f"不支持的数据维度: {ct_data.ndim}，期望3或4维")
    
    # 数据形状 (D, H, W) 对应 (z, y, x) 或某种排列
    # 假设数据格式为 (Z, Y, X) 或 (Z, X, Y)
    d, h, w = data.shape
    
    # 计算中心索引
    mid_d = d // 2
    mid_h = h // 2
    mid_w = w // 2
    
    # 提取三个正交切片
    slices = {
        'axial': data[mid_d, :, :],      # Z轴中心切片 (Y, X)
        'sagittal': data[:, :, mid_w],   # X轴中心切片 (Z, Y)
        'coronal': data[:, mid_h, :]     # Y轴中心切片 (Z, X)
    }
    
    return slices

def flip_ct_in_h5(input_path, output_path, flip_axis=1, inplace=False, verify=True):
    """
    对H5文件中的所有CT数据进行flip操作
    
    参数:
        input_path: 输入H5文件路径
        output_path: 输出H5文件路径
        flip_axis: 要翻转的维度（默认1，即第0维度如果数据是(1,64,128,128)的话）
        inplace: 是否原地修改（True）还是保存到新文件（False）
        verify: 是否验证翻转结果
    """
    
    input_path = Path(input_path)
    output_path = Path(output_path)
    
    if not input_path.exists():
        raise FileNotFoundError(f"输入文件不存在: {input_path}")
    
    # 如果是原地修改，先备份
    if inplace:
        backup_path = input_path.with_suffix('.h5.backup')
        print(f"创建备份: {backup_path}")
        shutil.copy2(input_path, backup_path)
        output_path = input_path
    else:
        # 确保输出目录存在
        output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 首先统计有多少个文件需要处理
    with h5py.File(input_path, 'r') as hf_in:
        total_files = len(list(hf_in.keys()))
    
    print(f"\n{'='*70}")
    print(f"开始处理 H5 文件")
    print(f"  输入文件: {input_path}")
    print(f"  输出文件: {output_path}")
    print(f"  Flip维度: axis={flip_axis}")
    print(f"  总文件数: {total_files}")
    print(f"{'='*70}\n")
    
    # 打开输入和输出文件
    with h5py.File(input_path, 'r') as hf_in:
        # 如果不是原地修改，创建新的H5文件
        if not inplace:
            with h5py.File(output_path, 'w') as hf_out:
                _process_h5_groups(hf_in, hf_out, flip_axis, verify)
        else:
            # 原地修改：先读取所有数据，然后重新写入
            _process_h5_inplace(hf_in, flip_axis, verify)
    
    print(f"\n{'='*70}")
    print(f"✓ 处理完成!")
    print(f"  输出文件: {output_path}")
    if inplace:
        print(f"  备份文件: {backup_path}")
    print(f"{'='*70}")

def _process_h5_groups(hf_in, hf_out, flip_axis, verify):
    """处理H5文件的所有组（保存到新文件）"""
    
    file_keys = list(hf_in.keys())
    
    for file_key in tqdm(file_keys, desc="处理文件"):
        grp_in = hf_in[file_key]
        grp_out = hf_out.create_group(file_key)
        
        # 处理CT数据
        if 'ct' in grp_in:
            ct_data = grp_in['ct'][:]
            
            # 显示原始形状（只显示第一个）
            if file_key == file_keys[0]:
                print(f"\n原始CT数据形状: {ct_data.shape}")
                print(f"翻转维度: axis={flip_axis}")
            
            # 执行flip
            ct_flipped = np.flip(ct_data, axis=flip_axis)
            
            # 验证（只验证第一个）
            if verify and file_key == file_keys[0]:
                print(f"翻转后形状: {ct_flipped.shape}")
                print(f"原始数据范围: [{ct_data.min():.2f}, {ct_data.max():.2f}]")
                print(f"翻转后范围: [{ct_flipped.min():.2f}, {ct_flipped.max():.2f}]")
                
                # 检查第一个和最后一个切片是否交换
                if ct_data.ndim >= 2:
                    axis_size = ct_data.shape[flip_axis]
                    original_first = np.take(ct_data, 0, axis=flip_axis)
                    flipped_last = np.take(ct_flipped, axis_size-1, axis=flip_axis)
                    if np.allclose(original_first, flipped_last):
                        print("✓ 验证成功: 第一个切片已翻转到最后")
                    else:
                        print("⚠ 警告: 翻转验证失败")
            
            # 保存翻转后的数据
            grp_out.create_dataset("ct", data=ct_flipped, compression="gzip", dtype="float32")
        
        # 复制其他数据集（保持不变）
        for key in grp_in.keys():
            if key != 'ct':
                grp_out.create_dataset(key, data=grp_in[key][:])
        
        # 复制属性
        for attr_key, attr_val in grp_in.attrs.items():
            grp_out.attrs[attr_key] = attr_val

def _process_h5_inplace(hf, flip_axis, verify):
    """原地修改H5文件（先读取所有数据再重写）"""
    
    file_keys = list(hf.keys())
    
    # 第一步：读取所有数据
    print("步骤1/2: 读取所有数据...")
    all_data = {}
    for file_key in tqdm(file_keys, desc="读取"):
        grp = hf[file_key]
        if 'ct' in grp:
            ct_data = grp['ct'][:]
            ct_flipped = np.flip(ct_data, axis=flip_axis)
            all_data[file_key] = ct_flipped
    
    # 第二步：重新写入数据
    print("\n步骤2/2: 写入翻转后的数据...")
    for file_key in tqdm(file_keys, desc="写入"):
        if file_key in all_data:
            grp = hf[file_key]
            
            # 删除旧的CT数据集
            if 'ct' in grp:
                del grp['ct']
            
            # 写入新的CT数据
            grp.create_dataset("ct", data=all_data[file_key], compression="gzip", dtype="float32")

def main():
    parser = argparse.ArgumentParser(
        description="对H5文件中的CT数据进行flip操作（支持三视图预览）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 1. 先预览前10个样本的flip效果（三视图）
  python flip_ct_h5.py --input data.h5 --preview --axis 1
  
  # 2. 预览前20个样本
  python flip_ct_h5.py --input data.h5 --preview --num_visualize 20 --axis 1
  
  # 3. 确认无误后，执行实际flip操作
  python flip_ct_h5.py --input data.h5 --output data_flipped.h5 --axis 1
  
  # 4. 原地修改（会创建备份）
  python flip_ct_h5.py --input data.h5 --inplace --axis 1

数据格式说明:
  假设CT数据形状为 (1, 64, 128, 128) 或 (64, 128, 128):
  - axis=0: 翻转通道维度 (通常不需要)
  - axis=1: 翻转Z轴 (64层面) - 常用于轴向翻转
  - axis=2: 翻转Y轴 (128行) - 常用于前后翻转
  - axis=3: 翻转X轴 (128列) - 常用于左右翻转
        """
    )
    parser.add_argument("--input", type=str, required=True,
                       help="输入H5文件路径")
    parser.add_argument("--output", type=str, default=None,
                       help="输出H5文件路径（默认为输入文件名_flipped.h5）")
    parser.add_argument("--axis", type=int, default=1,
                       help="要翻转的维度（默认1，对应(1,64,128,128)中的64层面）")
    parser.add_argument("--inplace", action="store_true",
                       help="原地修改输入文件（会先创建.backup备份）")
    parser.add_argument("--preview", action="store_true",
                       help="预览模式：只生成三视图对比，不修改文件")
    parser.add_argument("--num_visualize", type=int, default=10,
                       help="预览模式下要可视化的样本数量（默认10）")
    parser.add_argument("--preview_dir", type=str, default="flip_preview",
                       help="预览图输出目录（默认flip_preview）")
    
    args = parser.parse_args()
    
    # 预览模式
    if args.preview:
        visualize_flip_preview(
            input_path=args.input,
            flip_axis=args.axis,
            num_samples=args.num_visualize,
            output_dir=args.preview_dir
        )
        return
    
    # 实际flip操作模式
    # 确定输出路径
    if args.output is None and not args.inplace:
        input_path = Path(args.input)
        args.output = input_path.parent / f"{input_path.stem}_flipped.h5"
    
    # 执行flip操作
    flip_ct_in_h5(
        input_path=args.input,
        output_path=args.output if not args.inplace else args.input,
        flip_axis=args.axis,
        inplace=args.inplace,
        verify=True
    )

if __name__ == "__main__":
    main()