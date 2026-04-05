import pandas as pd
import glob
import os

def merge_csv_files():
    # 获取当前文件夹下所有csv文件
    csv_files = glob.glob('*.csv')
    
    if not csv_files:
        print("当前文件夹下没有找到CSV文件")
        return
    
    print(f"找到 {len(csv_files)} 个CSV文件:")
    for f in csv_files:
        print(f"  - {f}")
    
    # 读取所有CSV文件并合并
    dfs = []
    for i, file in enumerate(csv_files):
        try:
            # 读取CSV文件
            df = pd.read_csv(file)
            dfs.append(df)
            print(f"已读取: {file} ({len(df)} 行)")
        except Exception as e:
            print(f"读取 {file} 时出错: {e}")
    
    if not dfs:
        print("没有成功读取任何CSV文件")
        return
    
    # 合并所有dataframe
    merged_df = pd.concat(dfs, ignore_index=True)
    print(f"\n合并后总行数: {len(merged_df)}")
    
    # 按第一列升序排序
    first_column = merged_df.columns[0]
    merged_df = merged_df.sort_values(by=first_column, ascending=True)
    print(f"按第一列 '{first_column}' 升序排序完成")
    
    # 保存到All_ML.csv
    output_file = 'All_DL.csv'
    merged_df.to_csv(output_file, index=False)
    print(f"\n已保存到: {output_file}")
    print(f"最终文件包含 {len(merged_df)} 行数据")

if __name__ == "__main__":
    merge_csv_files()