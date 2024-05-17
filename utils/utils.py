from scapy.all import IP, IPv6
def match_keys(key1, key2):
    """
    检查两个键是否匹配。
    
    :param key1: 第一个键，字典形式，包含IP地址、协议，可选地包含端口号。
    :param key2: 第二个键，格式同key1。
    :return: 布尔值，True 如果两个键匹配，否则 False。
    """
    # 检查IP地址和协议是否匹配
    if key1 is None or key2 is None:
        return False
    if key1['protocol'] != key2['protocol']:
        return False
    ip11, ip12 = key1['ip']
    ip21, ip22 = key2['ip']
    if (ip11 == ip21 and ip12 == ip22) or (ip11 == ip22 and ip21 == ip12):
    # 如果两个键都有port字段
        if 'port' not in key1 and 'port' not in key2:
            return True
        elif 'port' in key1 and 'port' in key2:
            # 检查端口号是否匹配（不考虑方向）
            port1_src, port1_dst = key1['port']
            port2_src, port2_dst = key2['port']
            match_direct = (port1_src == port2_src and port1_dst == port2_dst)
            match_reverse = (port1_src == port2_dst and port1_dst == port2_src)
            return match_direct or match_reverse
    return False
def protocol_to_int(protocol):
    """
    将协议名称转换为整数。
    
    :param protocol: 协议名称，字符串。
    :return: 整数，对应的协议号。
    """
    if protocol == 'ICMP':
        return 1
    elif protocol == 'TCP':
        return 6
    elif protocol == 'UDP':
        return 17
    elif protocol == 'DNS':
        return 53
    elif protocol == 'NTP':
        return 123
    else:
        return None
def extract_ip(packet):
    """
    从数据包中提取源IP地址和目标IP地址。
    
    :param packet: 数据包对象。
    :return: 源IP地址和目标IP地址的元组。
    """
    if IP in packet:
        return packet[IP].src, packet[IP].dst
    elif IPv6 in packet:
        return packet[IPv6].src, packet[IPv6].dst
    else:
        return None, None
def compare(src_ip, dst_ip):
    """
    比较两个IP地址，返回它们的大小关系。
    
    :param src_ip: 源IP地址。
    :param dst_ip: 目标IP地址。
    :return: 整数，1 表示src_ip > dst_ip，0 表示相等，-1 表示src_ip < dst_ip。
    """
    src_str = str(src_ip).split('.')
    dst_str = str(dst_ip).split('.')
    for src, dst in zip(src_str, dst_str):
        if int(src) > int(dst):
            return 1
        elif int(src) < int(dst):
            return -1
if __name__ == '__main__':
    # 示例用法
    key1, key2 = "192.168.1.1", "192.168.1.2"
    print(compare(key1, key2))  # -1