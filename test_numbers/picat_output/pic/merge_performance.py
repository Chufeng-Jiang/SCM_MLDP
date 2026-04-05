import os
import csv
from pathlib import Path
from turtle import pd

def merge_and_sort_csv_files():
    """
    读取当前目录下所有CSV文件，合并并按第一列升序排序
    纯Python实现，不依赖pandas
    """
    # 获取当前目录
    current_dir = Path("../DP/")
    
    # 获取所有CSV文件（排除输出文件）
    csv_files = sorted([f for f in current_dir.glob("*.csv") 
                       if f.name != "merged_DP.csv"])
    
    if not csv_files:
        print("错误: 当前目录中没有找到CSV文件")
        return
    
    print(f"找到 {len(csv_files)} 个CSV文件:")
    for f in csv_files:
        print(f"  - {f.name}")
    print()
    
    # 存储所有数据行（保持原始字符串格式）
    all_rows = []
    header = None
    
    for i, csv_file in enumerate(csv_files):
        try:
            with open(csv_file, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                
                # 读取列名
                file_header = next(reader)
                
                if i == 0:
                    # 第一个文件，保存列名
                    header = file_header
                    print(f"列名: {', '.join(header)}")
                
                # 读取数据行（保持字符串格式）
                row_count = 0
                for row in reader:
                    if row and row[0].strip():  # 跳过空行
                        all_rows.append(row)
                        row_count += 1
                
                print(f"读取 {csv_file.name}: {row_count} 行")
                
        except Exception as e:
            print(f"警告: 无法读取 {csv_file.name}: {e}")
    
    if not all_rows:
        print("错误: 没有读取到任何数据")
        return
    
    print(f"\n合并后总行数: {len(all_rows)}")
    
    # 按第一列排序（作为整数比较）
    print(f"按第一列升序排序...")
    
    try:
        # 按整数值排序，但保持原始字符串格式
        all_rows.sort(key=lambda x: int(x[0]) if x[0].strip() else 0)
    except ValueError as e:
        print(f"警告: 无法按数值排序，改用字符串排序: {e}")
        all_rows.sort(key=lambda x: x[0])
    
    # 写入结果
    output_file = "DP.csv"
    print(f"正在写入到 {output_file}...")
    
    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        
        # 写入列名
        writer.writerow(header)
        
        # 写入所有数据行（保持原始格式）
        for row in all_rows:
            writer.writerow(row)
    
    print(f"\n✓ 合并完成!")
    print(f"  输出文件: {output_file}")
    print(f"  总行数: {len(all_rows)}")
    
    # 显示第一列的范围
    if all_rows:
        print(f"  {header[0]} 范围: {all_rows[0][0]} ~ {all_rows[-1][0]}")
        print(f"  第一个数字长度: {len(all_rows[0][0])} 位")
        print(f"  最后一个数字长度: {len(all_rows[-1][0])} 位")

if __name__ == "__main__":
    print("=" * 60)
    print("CSV 文件合并与排序工具（保持大数字精度）")
    print("=" * 60)
    print()
    
    merge_and_sort_csv_files()