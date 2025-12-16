import os  
import h5py  
import nibabel as nib  
import numpy as np  
import cv2  
import torchio as tio  
from multiprocessing import Pool, Manager, Lock  
from tqdm import tqdm  
import time 

def normalize_data(image, mask, clip_limit=2.0, tile_grid_size=(8, 8), window_min=-2000, window_max=1000):
    def adaptive_windowing(img, min_val=-2000, max_val=1000):
        img_filtered = np.array(img)
        img_filtered[img_filtered < img_filtered.min() + 100] = np.nan
        min_display = np.nanpercentile(img_filtered, 0.5)
        max_display = np.nanpercentile(img_filtered, 99.5)
        img_windowed = np.clip(img, min_display, max_display)
        img_windowed = ((img_windowed - min_display) / (max_display - min_display) * 255).astype(np.uint8)
        return img_windowed

    def apply_clahe(img, clip_limit=2.0, tile_grid_size=(8, 8)):
        clahe_slices = []
        for i in range(img.shape[0]):
            slice_img = img[i, :, :]
            clahe_img = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size).apply(slice_img)
            clahe_slices.append(clahe_img)
        return np.stack(clahe_slices, axis=0)

    def normalize(img):
        return img.astype(np.float32) / 255.0

    image_windowed = adaptive_windowing(image, window_min, window_max)
    image_clahe = apply_clahe(image_windowed, clip_limit, tile_grid_size)
    image_normalized = normalize(image_clahe)
    image_normalized = np.expand_dims(image_normalized, axis=0)
    return image_normalized, mask

def resize_data(image, mask, target_shape=(64, 128, 128)):
    resize_transform = tio.Resize(target_shape)
    image = resize_transform(image)
    mask = resize_transform(mask)
    return image, mask

def resample_data(image, mask, target_spacing=(1, 1, 1)):
    resample_transform = tio.Resample(target_spacing)
    image = resample_transform(image)
    mask = resample_transform(mask)
    return image, mask

def process_single_file(args):  
    """  
    处理单个文件的函数  
    """  
    file_name, ct_base_path, mask_base_path, output_path, region_names, process_lock = args  
    
    try:  
        # 查找CT文件  
        ct_file_path = None  
        for root, dirs, files in os.walk(ct_base_path):  
            for file in files:  
                if file == f"{file_name}.nii.gz":  
                    ct_file_path = os.path.join(root, file)  
                    break  
            if ct_file_path:  
                break  

        if not ct_file_path:  
            return f"CT文件 {file_name}.nii.gz 未找到，跳过"  

        # 加载CT图像  
        ct_img = nib.load(ct_file_path).get_fdata()  
        ct_img = np.transpose(ct_img, (2, 0, 1))  

        # 查找掩码文件  
        mask_folder = os.path.join(mask_base_path, f"seg_{file_name}")  
        if not os.path.isdir(mask_folder):  
            return f"掩码文件夹 seg_{file_name} 未找到，跳过"  

        masks = []  
        for region in region_names:  
            mask_file_path = os.path.join(mask_folder, f"{region}.nii.gz")  
            if os.path.exists(mask_file_path):  
                mask_img = nib.load(mask_file_path).get_fdata()  
                mask_img = np.transpose(mask_img, (2, 0, 1))  
                masks.append(mask_img.astype(bool))  
            else:  
                return f"掩码文件 {mask_file_path} 未找到，跳过"  

        if len(masks) != len(region_names):  
            return f"{file_name} 的掩码数量不足，跳过"  

        # 合成四维掩码数据  
        masks_4d = np.stack(masks, axis=0)  

        # 对数据进行预处理  
        ct_img, masks_4d = normalize_data(ct_img, masks_4d)  
        ct_img, masks_4d = resample_data(ct_img, masks_4d)  
        ct_img, masks_4d = resize_data(ct_img, masks_4d)  

        # 使用进程锁保护HDF5文件写入  
        with process_lock:  
            with h5py.File(output_path, "a") as h5f:  
                if file_name not in h5f:  
                    grp = h5f.create_group(file_name)  
                    grp.create_dataset("ct", data=ct_img.astype(np.float32), compression="gzip", dtype="float32")  
                    grp.create_dataset("mask", data=masks_4d, compression="gzip", dtype="bool")  

        return f"成功处理文件 {file_name}"  

    except Exception as e:  
        return f"处理文件 {file_name} 时发生错误: {str(e)}"  

def main():  
    # 读取已处理的文件  
    processed_files = set()  
    output_path = "/data/birth/lmx/work/Class_projects/bxg/valid_total_processed_data.h5"
    if os.path.exists(output_path):  
        with h5py.File(output_path, "r") as h5f:  
            processed_files = set(h5f.keys())  
    print(f"已找到{len(processed_files)}个处理过的文件")  

    # 定义处理的解剖区域
    region_names = ["lung", "trachea and bronchie", "pleura", "mediastinum", "heart",  
                   "esophagus", "bone", "thyroid", "abdomen"]  

    # 数据路径  
    ct_base_path = "/data/birth/lmx/work/Class_projects/bxg/VM-UNet/data/valid/valid_preprocessed"  
    mask_base_path = "/data/birth/lmx/work/Class_projects/bxg/VM-UNet/data/valid/valid_region_mask"  

    # 获取所有需要处理的文件名（不包括已处理的文件）
    all_files = set()
    for root, dirs, files in os.walk(ct_base_path):
        for file in files:
            if file.endswith('.nii.gz'):
                file_name = os.path.splitext(os.path.splitext(file)[0])[0]
                all_files.add(file_name)
    
    remaining_files = list(all_files - processed_files)

    print(f"总共发现{len(all_files)}个文件")
    print(f"已处理{len(processed_files)}个文件")
    print(f"剩余{len(remaining_files)}个文件待处理")  

    # 创建进程管理器和进程锁  
    manager = Manager()  
    process_lock = manager.Lock()  

    # 准备进程池参数  
    num_processes = 4  # 可以根据CPU核心数调整  
    args_list = [(file_name, ct_base_path, mask_base_path, output_path, region_names, process_lock)  
                 for file_name in remaining_files]  

    # 使用进程池处理文件  
    with Pool(processes=num_processes) as pool:  
        # 使用imap处理任务并显示进度条  
        for result in tqdm(pool.imap_unordered(process_single_file, args_list),  
                         total=len(args_list), desc="处理进度"):  
            print(result)  

    print(f"所有剩余数据已处理完成并保存到 {output_path}")  

if __name__ == "__main__":  
    main()