import re
import time
from datetime import datetime
import os
import sys
import pickle
import ipaddress
from src.RttTable import RTTTable
from src.Monitor import NetworkTrafficMonitor
import logging
def convert_to_decimal(timestamp_str):
    # 修改正则表达式以匹配新的时间格式
    pattern = r"(\w{3} \w{3} \d{2} \d{2}:\d{2}:\d{2}\.\d{6} \d{4})"
    match = re.match(pattern, timestamp_str)
    if match:
        cleaned_timestamp_str = match.group(1)
    else:
        raise ValueError("Invalid timestamp format")

    # 使用正确的格式解析时间字符串
    dt = datetime.strptime(cleaned_timestamp_str, "%a %b %d %H:%M:%S.%f %Y")

    # 将 datetime 对象转换为 Unix 时间戳
    unix_timestamp = time.mktime(dt.timetuple())

    # 添加微秒部分，转换为十进制形式的时间戳
    decimal_timestamp = unix_timestamp + dt.microsecond / 1e6
    return decimal_timestamp

def save_data_with_pickle(data, filename):
    with open(filename, 'wb') as file:
        pickle.dump(data, file)

def extract_rtt(file_name):
    with open(file_name, 'r') as file:
        lines = file.readlines()

    connections = []
    connection = {}
    for line in lines:
        if "host " in line:
            host_info = re.search(r'\s*host \w+: (.*)', line.strip()).group(1)
            ip_port = re.search(r'(.*):(\d+)', host_info).group(1, 2)
            if 'host_a' not in connection:
                connection['host_a'] = {'ip': ip_port[0].strip(), 'port': ip_port[1].strip()}
            else:
                connection['host_b'] = {'ip': ip_port[0].strip(), 'port': ip_port[1].strip()}
        elif 'packet' in line:
            time_match = re.search(r'(first packet|last packet):\s+(.*)', line.strip())
            if time_match:
                timestamp_str = time_match.group(2)
                if 'first_packet' not in connection:
                    connection['first_packet'] = convert_to_decimal(timestamp_str)
                else:
                    connection['last_packet'] = convert_to_decimal(timestamp_str)
        elif 'RTT samples' in line:
            rtt_samples = re.findall(r'(RTT samples):\s+(\d+)', line)
            for match in rtt_samples:
                key, value = match[0].lower().replace(' ', '_'), int(match[1])
                if key not in connection:
                    connection[key] = [value]
                else:
                    connection[key].append(value)
        else:
            matches = re.findall(r'(RTT samples|RTT avg|RTT min|RTT max):\s+(\d+\.\d+|\d+)', line)
            for match in matches:
                key, value = match[0].lower().replace(' ', '_'), float(match[1])
                if key not in connection:
                    connection[key] = [value]
                else:
                    connection[key].append(value)

        if 'host_b' in connection and 'rtt_avg' in connection:
            connections.append(connection)
            connection = {}

    return connections
def setup_logging(name='network_monitor'):
    '''
    params:
        name (str): The name of the logger.
    设置日志记录器。
    '''
    log_directory = "./logs"
    log_path = os.path.join(log_directory, f"{name}.log")
    print(log_path)
    if not os.path.exists(log_directory):
        os.makedirs(log_directory)

    logging.basicConfig(
        filename=log_path,
        filemode='a',
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    logger = logging.getLogger(name)
    return logger

def main(dir_path, output_dir):
    # 初始化计数器和 RTT 表
    count = 0
    tcptrace_table = RTTTable()
    
    # 设置日志记录
    logger = setup_logging('tcptrace')
    
    # 创建网络流量监控器实例，命名为 tcptrace，并关闭异常检测
    tcptrace_monitor = NetworkTrafficMonitor(name='tcptrace', check_anomalies=False, logger=logger)
    
    # 获取指定目录下的所有文件名并排序
    filenames = sorted(os.listdir(dir_path))
    
    # 遍历每个文件
    for filename in filenames:
        # 初始化最小和最大时间
        min_time = -1
        max_time = 0
        
        # 确保输出目录存在，不存在则创建
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 仅处理 .txt 文件
        if filename.endswith(".txt"):
            count += 1  # 更新处理文件计数器
            
            # 提取 RTT（往返时间）信息
            connections = extract_rtt(f"{dir_path}/{filename}")
            
            # 遍历每个连接
            for connection in connections:
                # 初始化连接数据字典，包含 RTT 采样和包信息
                value = {
                    'rtt_samples': connection['rtt_samples'],
                    'first_packet': connection['first_packet'],
                    'last_packet': connection['last_packet'],
                    'rtt_min': connection['rtt_min'],
                    'rtt_max': connection['rtt_max'],
                    'rtt_avg': connection['rtt_avg'],
                    'port_a': connection['host_a']['port'],
                    'port_b': connection['host_b']['port']
                }
                
                # 更新最小和最大时间
                if connection['first_packet'] < min_time or min_time == -1:
                    min_time = connection['first_packet']
                if connection['last_packet'] > max_time:
                    max_time = connection['last_packet']
                
                # 获取源和目标 IP 地址
                src_ip = connection['host_a']['ip']
                dst_ip = connection['host_b']['ip']
                
                # 确定传输方向
                dir, reversed_dir = 'forward', 'backward'
                if ipaddress.ip_address(src_ip) > ipaddress.ip_address(dst_ip):
                    # 如果源 IP 大于目标 IP，交换它们的顺序
                    src_ip, dst_ip = dst_ip, src_ip
                    dir, reversed_dir = reversed_dir, dir
                
                # 检查 RTT 样本是否存在，并处理第一个样本
                if connection['rtt_samples'][0] != 0:
                    value1 = {
                        'rtt': connection['rtt_min'][0],
                        'timestamp': connection['last_packet'],
                        'types': 'tcptrace'
                    }
                    # 将 RTT 样本添加到 RTT 表
                    tcptrace_table.add_rtt_sample(
                        src_ip=src_ip,
                        dst_ip=dst_ip,
                        rtt=value1['rtt'],
                        timestamp=value1['timestamp'],
                        types='tcptrace',
                        direction=dir
                    )
                    # 更新 RTT 信息到监控器
                    tcptrace_monitor.add_or_update_ip_with_rtt(
                        dst_ip if dir == 'forward' else src_ip,
                        'TCP',
                        'tcptrace',
                        value1['rtt'],
                        timestamp=value1['timestamp']
                    )
                
                # 处理第二个样本
                if connection['rtt_samples'][1] != 0:
                    value2 = {
                        'rtt': connection['rtt_min'][1],
                        'timestamp': connection['last_packet'],
                        'types': 'tcptrace'
                    }
                    tcptrace_table.add_rtt_sample(
                        src_ip=src_ip,
                        dst_ip=dst_ip,
                        rtt=value2['rtt'],
                        timestamp=value2['timestamp'],
                        types='tcptrace',
                        direction=reversed_dir
                    )
                    tcptrace_monitor.add_or_update_ip_with_rtt(
                        src_ip if reversed_dir == 'backward' else dst_ip,
                        'TCP',
                        'tcptrace',
                        value2['rtt'],
                        timestamp=value2['timestamp']
                    )
                
                # 更新 RTT 表的最大和最小时间
                tcptrace_table.max_time = max_time
                tcptrace_table.min_time = min_time
                
                # 进行连接的测量结果与 baseline/tcptrace 的差异比较（注释中标出未实现）
                
        else:
            continue  # 跳过非 .txt 文件

    # 使用 pickle 序列化并保存 RTT 表和监控器到输出目录
    save_data_with_pickle(tcptrace_table, os.path.join(output_dir, 'new_tcptrace.pkl'))
    save_data_with_pickle(tcptrace_monitor, os.path.join(output_dir, 'new_tcptrace_monitor.pkl'))
    
    # 将 RTT 表和监控器的内容输出到文本文件
    with open(os.path.join(output_dir, 'new_tcptrace.txt'), 'w') as file:
        sys.stdout = file
        tcptrace_table.print_tcprtt()  # 打印 RTT 表
        tcptrace_monitor.print_trees()  # 打印监控器的 IP 树结构
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <input_directory> <output_directory>")
        sys.exit(1)

    input_directory = sys.argv[1]
    output_directory = sys.argv[2]
    main(input_directory, output_directory)