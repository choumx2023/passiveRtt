#!/bin/bash

# ========================
# 参数检查与初始化
# ========================
if [ "$#" -ne 5 ]; then
    echo "Usage: $0 <input_pcap_file> <output_directory> <output_data_dir> <final_result> <time_slot>"
    exit 1
fi

input_file="$1"
out_dir="$2"
data_dir="$3"
final_result="$4"
time_slot="$5"

split_pcap_dir="${out_dir}/split_pcap"
split_rtt_dir="${out_dir}/split_rtt"

mkdir -p "$split_pcap_dir" "$split_rtt_dir"

# ========================
# 主流程：PCAP 分割与处理
# ========================
echo "[1/3] Splitting PCAP by $time_slot seconds..."
python3 split_pcap.py "$input_file" "$split_pcap_dir" "output2" "$time_slot"

echo "[2/3] Running tcptrace on split PCAPs..."
for pcap_file in "$split_pcap_dir"/*.pcap; do
    base_name=$(basename "$pcap_file" .pcap)
    echo "  → Processing $base_name"
    tcptrace -n -r -l --output_dir="$out_dir" --output_prefix="${base_name}_analysis" "$pcap_file" \
        > "$split_rtt_dir/${base_name}.txt" 2>&1
done

echo "[3/3] Post-processing tcptrace results..."
python3.12 tcptrace_process.py "$split_rtt_dir" "$out_dir"
echo "✅ All processing complete."

# ========================
# 可选后处理（按需取消注释）
# ========================
# python3 current_process.py "$input_file" "$data_dir"
# echo "Current process complete."

# python3 ./analysis/rtt_relay.py "$data_dir" "$out_dir" "$final_result" --timeslot="$time_slot"
# echo "RTT relay process done."

# 临时文件清理
# rm -rf "$split_pcap_dir" "$split_rtt_dir"