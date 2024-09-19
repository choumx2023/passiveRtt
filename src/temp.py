
def defalute_state():
    return {
        'forward_range' : [-1, -1],
        'backward_range': [-1, -1],
        'forward_sack_range' : [-1, -1],
        'backward_sack_range': [-1, -1],
        'time_series': []
        # 重传：
        # 我发对面没发：我确认的不变，但是我放最远的发
        # 对面发我没发：我的序号不变，但是我确认对面的
        # 我发对面发： 自己控制的同时更新，并且不超过对面的最大值
        # 丢包：非常规的确认 比如出现了我确认了比对面最远的还遥远的数据
    }


MAX_SEQ = 0xFFFFFFFF  # 4294967295, TCP最大序列号

def seq_compare(seq1, seq2):
    """
    比较两个序列号，考虑回环。
    如果 seq1 在 seq2 之后，返回 True。
    """
    if seq2 == -1: # 未初始化
        return True
    return (seq1 - seq2) % (MAX_SEQ + 1) < (MAX_SEQ + 1) // 2


def calculate_tcp_states(packet : dict):
    check = 0
    states = defalute_state()
    seq = packet['seq']
    ack = packet['ack']
    len = packet['len']
    is_syn = packet['syn'] == 1
    is_ack = packet['ack'] == 1
    furthest = seq + len - 1 + is_syn
    nearest = ack - 1
    judge = 0
    if packet['direction'] == 'forward':
        if seq_compare(furthest, states['forward_range'][0]):
            states['forward_range'][0] = furthest
            judge += 1
        if seq_compare(nearest, states['backward_range'][0]):
            states['backward_range'][0] = nearest
            judge += 2
        if judge == 0: # 没有更新 需要判定是重传还是心跳
            if is_syn and is_ack:
                states['time_series'].append('syn-ack')
            elif is_syn:
                states['time_series'].append('syn')
            elif len == 0:
                states['time_series'].append('forward heartbeat')
            else:
                states['time_series'].append('forward retransmission')
        elif judge == 1: # 只更新了自己的further range
            states['time_series'].append('forward back to back')
        elif judge == 2: # 只更新了对面的nearest range
            states['time_series'].append('forward ack')
        else: # 同时更新了两个range
            states['time_series'].append('forward normal')
    
    else:
        if seq_compare(furthest, states['backward_range'][0]):
            states['backward_range'][0] = furthest
            judge += 1
        if seq_compare(nearest, states['forward_range'][0]):
            states['forward_range'][0] = nearest
            judge += 2
        if judge == 0:
            if is_syn and is_ack:
                states['time_series'].append('syn-ack')
            elif is_syn:
                states['time_series'].append('syn')
            elif len == 0:
                states['time_series'].append('backward heartbeat')
            else:
                states['time_series'].append('backward retransmission')
        elif judge == 1:
            states['time_series'].append('backward back to back')
        elif judge == 2:
            states['time_series'].append('backward ack')
        else:
            states['time_series'].append('backward normal')
        
            