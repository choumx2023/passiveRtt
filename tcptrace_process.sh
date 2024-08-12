#!/bin/bash

# 检查参数数量是否正确
if [ "$#" -ne 5 ]; then
    echo "Usage: $0 <input_pcap_file> <output_directory> <output_data_dir> <final_result> [--timeslot]"
    exit 1
fi

# 获取传递的参数
input_file="$1"
output_directory="$2"
output_prefix="output2"
output_dir="${output_directory}/split_pcap"
output_dir2="${output_directory}/split_rtt"
output_data_dir="$3"
final_result="$4"
time_slot="$5"


# 检查并创建输出目录
mkdir -p "$output_dir"

# 使用 editcap 按每秒一个文件的方式切分PCAP文件
/Applications/Wireshark.app/Contents/MacOS/editcap -i "$time_slot" "$input_file" "${output_dir}/${output_prefix}_temp.pcap"

# 获取生成的文件列表
files=(${output_dir}/${output_prefix}_temp*.pcap)
index=0
echo "${output_dir}/${output_prefix}_temp.pcap"
# 重命名文件
for file in "${files[@]}"; do
    formatted_index=$(printf "%03d" $index)
    mv "$file" "${output_dir}/${output_prefix}_${formatted_index}.pcap"
    ((index++))
done

echo "PCAP文件已成功按整数秒切分，并重命名完成，文件保存在${output_dir}目录中。"

# 设置包含切分后 PCAP 文件的目录


# 检查输出目录是否存在，如果不存在则创建
mkdir -p "$output_dir2"

# 遍历目录中的所有 pcap 文件
for pcap_file in "$output_dir"/*.pcap
do
    echo "Analyzing: $pcap_file"
    # 提取文件名，用于输出文件命名
    base_name=$(basename "$pcap_file" .pcap)
    # 运行 tcptrace 并将标准输出和错误输出重定向到一个文本文件
    tcptrace -n -r -l --output_dir="$output_directory" --output_prefix="${base_name}_analysis" "$pcap_file" > "$output_dir2/${base_name}.txt" 2>&1
done

echo "All files have been analyzed."

 
# 运行 Python 脚本
#  editcap -A "2023-04-11 23:45:00" -B "2023-04-11 23:45:20" 202304120045.pcap test5.pcap

# ./tcptrace_process.sh socurce_pcap tcp_output_dir current_output_dir analysis_output_dir timeslot  
# ./tcptrace_process.sh ./test/test2.pcap ./tcptrace/tcpresult ./current/result output/test2 0.2
# ./tcptrace_process.sh ./test/test5.pcap ./tcptrace/tcpresult5 ./current/result5 output/test5 0.2

python3.12 tcptrace_process.py "$output_dir2" "$output_directory" # 在output_directory中生成一个关键的pkl文件
echo 'tcp process done'
#python3 current_process.py "$input_file" "$output_data_dir" # 在output_data_dir中生成一个关键的pkl文件
#echo 'current process done'
# python3 ./analysis/rtt_relay.py "$output_data_dir"  "$output_directory" "$final_result" --timeslot="$time_slot" # 需要先读取output_data_dir中的pkl文件，然后读取
#echo 'rtt_relay done'
#rm -rf "$output_dir"
#rm -rf "$output_dir2"