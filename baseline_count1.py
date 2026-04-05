def count_ones_in_binary(decimals_file):
    """
    读取包含十进制数字的文件，统计每个数字二进制表示中1的个数
    
    参数:
        decimals_file: 包含十进制数字的文件路径
    """
    try:
        with open(decimals_file, 'r') as file:
            line_number = 0
            
            print("十进制\t二进制\t\t1的个数")
            print("-" * 30)
            
            for line in file:
                line_number += 1
                line = line.strip()  # 去除空白字符
                
                if not line:  # 跳过空行
                    continue
                
                try:
                    # 转换为整数
                    decimal_num = int(line)
                    
                    # 转换为二进制字符串（去掉'0b'前缀）
                    binary_str = bin(decimal_num)[2:]
                    
                    # 统计1的个数
                    ones_count = binary_str.count('1')
                    
                    # 输出结果
                    print(f"{decimal_num}\t{binary_str}\t\t{ones_count}")
                    
                except ValueError:
                    print(f"第 {line_number} 行不是有效的十进制数字: '{line}'")
                    
    except FileNotFoundError:
        print(f"错误: 找不到文件 '{decimals_file}'")
    except Exception as e:
        print(f"发生错误: {str(e)}")

def main():
    # 文件路径
    input_file = "decimals.txt"
    
    # 调用函数处理文件
    count_ones_in_binary(input_file)

if __name__ == "__main__":
    main()