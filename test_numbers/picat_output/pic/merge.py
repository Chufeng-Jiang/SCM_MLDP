import pandas as pd
import os
from pathlib import Path

def merge_csv_files(folder_path='.', output_file='merged_output.csv'):
    """
    合并当前文件夹下所有CSV文件，基于Number列
    
    参数:
        folder_path: 文件夹路径，默认为当前目录
        output_file: 输出文件名
    """
    # 获取所有CSV文件
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    
    if not csv_files:
        print("未找到CSV文件")
        return
    
    print(f"找到 {len(csv_files)} 个CSV文件:")
    for f in csv_files:
        print(f"  - {f}")
    
    # 读取第一个文件作为基础
    first_file = csv_files[0]
    merged_df = pd.read_csv(os.path.join(folder_path, first_file))
    print(f"\n开始合并，基础文件: {first_file}")
    print(f"  形状: {merged_df.shape}")
    
    # 逐个合并其他文件
    for csv_file in csv_files[1:]:
        file_path = os.path.join(folder_path, csv_file)
        df = pd.read_csv(file_path)
        print(f"\n合并文件: {csv_file}")
        print(f"  形状: {df.shape}")
        
        # 基于Number列进行外连接合并
        merged_df = pd.merge(
            merged_df, 
            df, 
            on='Number', 
            how='outer',
            suffixes=('', f'_{csv_file.replace(".csv", "")}')
        )
        print(f"  合并后形状: {merged_df.shape}")
    
    # 按Number列排序
    merged_df = merged_df.sort_values('Number').reset_index(drop=True)
    
    # 保存合并后的文件
    merged_df.to_csv(output_file, index=False)
    print(f"\n合并完成！")
    print(f"输出文件: {output_file}")
    print(f"最终形状: {merged_df.shape}")
    print(f"总共 {len(merged_df)} 个唯一数字")
    print(f"\n列名: {list(merged_df.columns)}")
    
    # 显示前几行
    print("\n前5行预览:")
    print(merged_df.head())
    
    return merged_df

def merge_specific_csv_files(file_list, output_file='merged_output.csv'):
    """
    合并指定的CSV文件列表
    
    参数:
        file_list: CSV文件路径列表
        output_file: 输出文件名
    """
    if not file_list:
        print("文件列表为空")
        return
    
    print(f"准备合并 {len(file_list)} 个文件:")
    for f in file_list:
        print(f"  - {f}")
    
    # 读取第一个文件
    merged_df = pd.read_csv(file_list[0])
    print(f"\n基础文件: {file_list[0]}")
    print(f"  形状: {merged_df.shape}")
    
    # 合并其他文件
    for file_path in file_list[1:]:
        df = pd.read_csv(file_path)
        print(f"\n合并文件: {file_path}")
        print(f"  形状: {df.shape}")
        
        # 基于Number列合并
        merged_df = pd.merge(
            merged_df, 
            df, 
            on='Number', 
            how='outer',
            suffixes=('', f'_{Path(file_path).stem}')
        )
        print(f"  合并后形状: {merged_df.shape}")
    
    # 排序并保存
    merged_df = merged_df.sort_values('Number').reset_index(drop=True)
    merged_df.to_csv(output_file, index=False)
    
    print(f"\n合并完成！")
    print(f"输出文件: {output_file}")
    print(f"最终形状: {merged_df.shape}")
    print(f"\n前5行预览:")
    print(merged_df.head())
    
    return merged_df

def append_csv_files(file_list, output_file='appended_output.csv'):
    """
    将多个CSV文件纵向追加（不基于Number合并，直接堆叠）
    
    参数:
        file_list: CSV文件路径列表
        output_file: 输出文件名
    """
    if not file_list:
        print("文件列表为空")
        return
    
    print(f"准备追加 {len(file_list)} 个文件:")
    
    df_list = []
    for file_path in file_list:
        df = pd.read_csv(file_path)
        print(f"  - {file_path}: {df.shape[0]} 行")
        df_list.append(df)
    
    # 纵向拼接
    appended_df = pd.concat(df_list, ignore_index=True)
    
    # 按Number排序（如果需要）
    if 'Number' in appended_df.columns:
        appended_df = appended_df.sort_values('Number').reset_index(drop=True)
    
    # 保存
    appended_df.to_csv(output_file, index=False)
    
    print(f"\n追加完成！")
    print(f"输出文件: {output_file}")
    print(f"总行数: {len(appended_df)}")
    print(f"\n前5行预览:")
    print(appended_df.head())
    
    return appended_df

if __name__ == "__main__":
    # 示例1: 合并当前文件夹下所有CSV文件（基于Number列合并）
    merge_csv_files('.', 'merged_all.csv')
    