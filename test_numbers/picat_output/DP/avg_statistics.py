import pandas as pd
import numpy as np
from pathlib import Path
import re

def extract_file_prefix(filename):
    """提取文件名的前两位数字"""
    match = re.search(r'^\d{2}', filename)
    return match.group() if match else filename[:2]

def process_csv_files():
    # 获取当前文件夹下所有CSV文件
    csv_files = list(Path('.').glob('*.csv'))
    
    if not csv_files:
        print("当前文件夹下没有找到CSV文件")
        return
    
    results = []
    
    for csv_file in csv_files:
        print(f"处理文件: {csv_file.name}")
        
        # 读取CSV文件
        df = pd.read_csv(csv_file)
        
        # 提取文件名前两位数字
        file_prefix = extract_file_prefix(csv_file.name)
        
        # 初始化结果字典
        row_data = {'file_prefix': file_prefix}
        
        # 跳过第一列，从第二列开始处理
        columns_to_process = df.columns[1:]
        
        # 对每个数值列计算统计信息
        for col in columns_to_process:
            # 尝试转换为数值类型
            try:
                numeric_col = pd.to_numeric(df[col], errors='coerce')
                
                # 如果列有有效的数值数据
                if numeric_col.notna().any():
                    mean_value = numeric_col.mean()
                    
                    # 单位转换处理
                    if 'DPTime(ms)' in col:
                        # 毫秒转秒
                        mean_value = mean_value / 1000
                        col_name = col.replace('DPTime(ms)', 'DPTime(s)')
                    elif 'DPTableBytes' in col:
                        # 字节转MB
                        mean_value = mean_value / (1024 * 1024)
                        col_name = col.replace('DPTableBytes', 'DPTable(MB)')
                    else:
                        col_name = col
                    
                    # 保留3位小数
                    row_data[f'{col_name}_mean'] = round(mean_value, 3)
                    
            except Exception as e:
                print(f"  列 '{col}' 处理出错: {e}")
                continue
        
        results.append(row_data)
    
    # 创建结果DataFrame
    result_df = pd.DataFrame(results)
    
    # 按file_prefix降序排序
    result_df = result_df.sort_values('file_prefix', ascending=True).reset_index(drop=True)
    
    # 保存到CSV，保留3位小数格式
    output_file = 'statistics_summary.csv'
    result_df.to_csv(output_file, index=False, float_format='%.3f')
    print(f"\n统计结果已保存到: {output_file}")
    print(f"处理了 {len(csv_files)} 个文件")
    
    return result_df

if __name__ == "__main__":
    result = process_csv_files()
    if result is not None:
        print("\n结果预览:")
        print(result.head())