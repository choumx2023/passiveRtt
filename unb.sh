#!/bin/bash


# 设置文件夹路径
directory="./test/Attacks"
# 设置输出目录的根路径
base_output_dir="./current/Attacks"

# 遍历文件夹中的所有.pcap文件
for file in "$directory"/*.pcap
do
  echo "Processing $file"

  # 提取文件名（不包括扩展名）
  filename=$(basename "$file" .pcap)

  # 创建该文件的特定输出目录
  output_dir="$base_output_dir/$filename"
  mkdir -p "$output_dir"

  # 调用Python脚本，传递当前文件和特定的输出目录
  python3 current_process.py "$file" "$output_dir"
done


# 设置文件夹路径
directory="./test/Attacks-2"
# 设置输出目录的根路径
base_output_dir="./current/Attacks-2"

# 遍历文件夹中的所有.pcap文件
for file in "$directory"/*.pcap
do
  echo "Processing $file"

  # 提取文件名（不包括扩展名）
  filename=$(basename "$file" .pcap)

  # 创建该文件的特定输出目录
  output_dir="$base_output_dir/$filename"
  mkdir -p "$output_dir"

  # 调用Python脚本，传递当前文件和特定的输出目录
  python3 current_process.py "$file" "$output_dir"
done


directory="./test/Benign"
# 设置输出目录的根路径
base_output_dir="./current/Benign"

# 遍历文件夹中的所有.pcap文件
for file in "$directory"/*.pcap
do
  echo "Processing $file"

  # 提取文件名（不包括扩展名）
  filename=$(basename "$file" .pcap)

  # 创建该文件的特定输出目录
  output_dir="$base_output_dir/$filename"
  mkdir -p "$output_dir"

  # 调用Python脚本，传递当前文件和特定的输出目录
  python3 current_process.py "$file" "$output_dir"
done


directory="./test/Benign-2"
# 设置输出目录的根路径
base_output_dir="./current/Benign-2"

# 遍历文件夹中的所有.pcap文件
for file in "$directory"/*.pcap
do
  echo "Processing $file"

  # 提取文件名（不包括扩展名）
  filename=$(basename "$file" .pcap)

  # 创建该文件的特定输出目录
  output_dir="$base_output_dir/$filename"
  mkdir -p "$output_dir"

  # 调用Python脚本，传递当前文件和特定的输出目录
  python3 current_process.py "$file" "$output_dir"
done


echo "All files processed."