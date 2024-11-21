import numpy as np 
def defalute_state():
    return {
        'forward_range' : [-1, -1], # A -> B : [acked, sent]
        'backward_range': [-1, -1], # B -> A : [acked, sent]
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
        if seq_compare(furthest, states['forward_range'][1]):
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
            states['time_series'].append('forward back to back candidate')
        elif judge == 2: # 只更新了对面的nearest range
            states['time_series'].append('forward ack')
        else: # 同时更新了两个range
            states['time_series'].append('forward normal')
    # 如果是反向的包
    else:
        # 使用seq_compare函数判断是否更新反方向发来的最远的字节序号
        if seq_compare(furthest, states['backward_range'][1]):
            states['backward_range'][0] = furthest
            judge += 1
        # 使用seq_compare函数判断是否更新正向发来的最近的字节序号
        if seq_compare(nearest, states['forward_range'][0]):
            states['forward_range'][0] = nearest
            judge += 2
        # 如果judge == 1, 说明他没有更新反方向最远和正方向最近的字节序号，说明是syn包、heartbeat包或者重传包
        if judge == 0:
            if is_syn and is_ack:
                states['time_series'].append('syn-ack')
            elif is_syn:
                states['time_series'].append('syn')
            elif len == 0:
                states['time_series'].append('backward heartbeat')
            else:
                states['time_series'].append('backward retransmission')
        # 如果judge == 1, 说明他更新了反方向最远的字节序号，但是没有更新正方向最近的字节序号，有可能是背靠背
        elif judge == 1:
            states['time_series'].append('backward back to back candidate')
        # 如果judge == 2, 说明他更新了正方向最近的字节序号，但是没有更新反方向最远的字节序号，可能是纯ack包
        elif judge == 2:
            states['time_series'].append('backward ack')
        # 如果同时更新了两个range，说明是正常的数据包
        else:
            states['time_series'].append('backward normal')
        class WelfordVariance:
    '''
    This class implements the Welford algorithm for calculating the variance of a stream of data points.
    It handles an initial buffer of data points that are not used in anomaly detection until the buffer is filled.
    '''
    def __init__(self, time_window: int = 1200, max_count: int = 120, initial_limit: int = 10):
        '''
        params:
            time_window: 时间窗口，用于限制数据点的时间范围。
            max_count: 最大计数，用于限制数据点的数量。
            initial_limit: 初始数据点限制，这些数据点将不用于异常检测直到它们的数量超过此限制。
        '''
        self.count = 0
        self.mean = 0
        self.M2 = 0
        self.time_window = time_window
        self.max_count = max_count
        self.initial_limit = initial_limit
        self.initial_count = 0
        self.data = deque()
    def update(self, x: float, timestamp: float):
        '''
        添加新的数据点并更新统计数据。
        '''
        self.data.append((x, timestamp))
        # 只有当数据点数量超过初始限制时，才更新统计信息
        if self.initial_count >= self.initial_limit:
            self.count += 1
            delta = x - self.mean
            self.mean += delta / self.count
            delta2 = x - self.mean
            self.M2 += delta * delta2
        else:
            self.initial_count += 1

    def remove(self, x: float):
        '''
        移除指定的数据点并更新统计数据。
        '''
        if self.count <= 1:
            self.count = 0
            self.mean = 0
            self.M2 = 0
        else:
            self.count -= 1
            delta = x - self.mean
            self.mean -= delta / self.count
            delta2 = x - self.mean
            self.M2 -= delta * delta2

    def variance(self):
        '''
        返回当前的方差。
        '''
        if self.count < 2:
            return float('nan')
        return self.M2 / (self.count - 1)

    def remove_outdated(self, current_time: float):
        '''
        移除超出时间窗口的数据点。
        '''
        while self.data and (current_time - self.data[0][1]) > self.time_window:
            old_value, _ = self.data.popleft()
            if self.initial_count >= self.initial_limit:
                self.remove(old_value)
            else:
                self.initial_count -= 1

    def check_anomalies(self, newrtt: float, timestamp: float):
        '''
        检查新数据点是否异常，并移除过时数据。
        '''
        self.remove_outdated(timestamp)

        if self.count >= 6 and newrtt - self.mean > min(20, max(15, 0.95 * math.sqrt(self.variance()))):
            return True
        
        self.update(newrtt, timestamp)
        return False
    def str_variance(self):
        if self.initial_count < self.initial_limit:
            return 'Not enough data points for variance calculation'
        if self.count < 2:
            return 'Not enough data points for variance calculation'
        variance = self.variance()
        std_dev = math.sqrt(variance) if not math.isnan(variance) else 0
        lower_bound = max(self.mean - std_dev, 0)
        upper_bound = self.mean + std_dev
        
        return f'Count: {self.count}, Variance: {variance}, Mean: {self.mean}, Chebyshev(1 sigma): [{lower_bound}, {upper_bound}]'




from collections import deque
import math

class WelfordVariance:
    def __init__(self, time_window: int = 1200, max_count: int = 120, initial_limit: int = 6):
        self.count = 0
        self.mean = 0
        self.M2 = 0
        self.time_window = time_window
        self.max_count = max_count
        self.initial_limit = initial_limit
        self.initial_data = deque()  # 存储前六个数据点
        self.data = deque()  # 存储所有数据点

    def update(self, x: float, timestamp: float):
        self.data.append((x, timestamp))

        # 更新所有数据点的统计信息
        self.count += 1
        delta = x - self.mean
        self.mean += delta / self.count
        delta2 = x - self.mean
        self.M2 += delta * delta2

        # 仅存储前六个数据点
        if len(self.initial_data) < self.initial_limit:
            self.initial_data.append((x, delta, delta2))

    def variance(self):
        if self.count - len(self.initial_data) < 2:
            return float('nan')
        
        adjusted_count = self.count - len(self.initial_data)
        adjusted_M2 = self.M2
        adjusted_mean = self.mean

        # 从M2中移除前六个数据点的贡献
        for x, delta, delta2 in self.initial_data:
            adjusted_mean -= delta / adjusted_count
            adjusted_M2 -= delta * delta2
            adjusted_count -= 1

        return adjusted_M2 / (adjusted_count - 1)

    def remove_outdated(self, current_time: float):
        while self.data and (current_time - self.data[0][1]) > self.time_window:
            old_value, _ = self.data.popleft()
            self.remove(old_value)

    def remove(self, x: float):
        if self.count <= 1:
            self.reset()
        else:
            self.count -= 1
            delta = x - self.mean
            self.mean -= delta / self.count
            delta2 = x - self.mean
            self.M2 -= delta * delta2
            if self.initial_data and x == self.initial_data[0][0]:
                self.initial_data.popleft()

    def reset(self):
        self.count = 0
        self.mean = 0
        self.M2 = 0
        self.initial_data.clear()

    def str_variance(self):
        return f'Count: {self.count - len(self.initial_data)}, Variance: {self.variance()}, Mean: {self.mean}'
    

import numpy as np

# 数据
data = [11, 12, 13, 14, 15, 60, 60, 60, 10, 11, 12]
window_size = 5  # 滑动窗口大小
threshold = 3    # 阈值倍数
delay_window = 3 # 延迟观察窗口

def delayed_anomaly_detection(data, window_size, threshold, delay_window):
    baseline = []
    results = []
    for i in range(len(data)):
        if len(baseline) < window_size:
            baseline.append(data[i])
            results.append((i, data[i], 'normal'))  # 前期直接加入基线
        else:
            mean = np.mean(baseline)
            std = np.std(baseline)
            # 判断是否为疑似异常
            if abs(data[i] - mean) > threshold * std:
                # 延迟观察后续数据
                if i + delay_window < len(data):
                    future_values = data[i+1:i+1+delay_window]
                    future_mean = np.mean(future_values)
                    # 如果后续数据趋于当前点，则认为趋势改变
                    if abs(data[i] - future_mean) < threshold * std:
                        baseline.append(data[i])
                        baseline = baseline[-window_size:]  # 保持窗口大小
                        results.append((i, data[i], 'trend_change'))
                    else:
                        results.append((i, data[i], 'anomaly'))
                else:
                    results.append((i, data[i], 'unknown'))
            else:
                baseline.append(data[i])
                baseline = baseline[-window_size:]  # 保持窗口大小
                results.append((i, data[i], 'normal'))
    return results

# 检测突变
results = delayed_anomaly_detection(data, window_size, threshold, delay_window)
for result in results:
    print(f"Index: {result[0]}, Value: {result[1]}, Status: {result[2]}")
    
    from collections import deque
from datetime import datetime, timedelta
import numpy as np
import math
def kmeans_1d(values, k=2, max_iter=10):
    """
    简化的一维 K-means 聚类算法。

    参数：
    - values: 一维数据列表
    - k: 聚类数
    - max_iter: 最大迭代次数

    返回：
    - centers: 聚类中心列表
    - labels: 每个数据点的聚类标签列表
    """
    # 初始化聚类中心，随机选择数据点
    centers = [values[i] for i in range(k)]
    for _ in range(max_iter):
        clusters = [[] for _ in range(k)]
        labels = []
        # 分配数据点到最近的聚类中心
        for v in values:
            distances = [abs(v - c) for c in centers]
            min_idx = distances.index(min(distances))
            clusters[min_idx].append(v)
            labels.append(min_idx)
        # 更新聚类中心
        new_centers = []
        for cluster in clusters:
            if cluster:
                new_centers.append(sum(cluster) / len(cluster))
            else:
                # 如果聚类为空，保持原有中心
                new_centers.append(centers[clusters.index(cluster)])
        # 检查是否收敛
        if new_centers == centers:
            break
        centers = new_centers
    return centers, labels
def calculate_within_group_distance(values, centers, labels):
    total_distance = [0, 0]
    counts = [0, 0]
    for v, l in zip(values, labels):
        total_distance[l] += abs(v - centers[l])**2
        counts[l] += 1
    return [math.sqrt(total_distance[l] / (counts[l] + 1e-8))  for l in range(2)]
def detect_changes_with_kmeans(data, time_window, center_change_threshold=0.2, cumulative_change_threshold=0.5):
    """
    使用 K-means 聚类方法检测时序数据中的突变和逐渐上升变化。

    参数：
    - data: 时序数据列表，格式为 [(timestamp, value), ...]
    - time_window: 时间窗口大小，timedelta 对象
    - center_change_threshold: 聚类中心单次变化的阈值比例
    - cumulative_change_threshold: 聚类中心累计变化的阈值比例

    返回：
    - results: 包含每个数据点的状态列表，元素格式为 (index, timestamp, value, status)
      - status: 'normal', 'sudden_change', 'gradual_increase'
    """
    window = deque()  # 时间窗口，存储 (timestamp, value)
    results = []
    prev_centers = None  # 前一个时间窗口的聚类中心
    cumulative_center_change = 0  # 累积的聚类中心变化
    centers = None
    weights = None
    distance = None
    for i, (timestamp, value) in enumerate(data):
        # 添加新数据点到窗口
        window.append((timestamp, value))

        # 移除超出时间窗口的数据点
        while window and (timestamp - window[0][0]) > time_window:
            window.popleft()

        # 当窗口中有足够的数据点时进行聚类
        if len(window) >= 2:
            values_in_window = [val for _, val in window]
            centers, labels = kmeans_1d(values_in_window, k=2)
            distance = calculate_within_group_distance(values_in_window, centers, labels)
            weights = [len([l for l in labels if l == 0]) / len(labels), len([l for l in labels if l == 1]) / len(labels)]
            centers.append(float(np.mean(values_in_window))) # 添加均值作为第三个聚类中心
            centers.append(centers[labels[-1]]) # 添加最后一个数据点作为第四个聚类中心
            status = 'normal'

            if prev_centers is not None:
                # 计算聚类中心的相对变化
                center_changes = [abs(c - pc) / (abs(pc) + 1e-8) for c, pc in zip(centers, prev_centers)]
                max_change = max(center_changes)

                cumulative_center_change += max_change

                if max_change > center_change_threshold:
                    status = 'sudden_change'
                    cumulative_center_change = 0  # 重置累积变化
                elif cumulative_center_change > cumulative_change_threshold:
                    status = 'gradual_increase'
                    cumulative_center_change = 0  # 重置累积变化

            # 更新前一个窗口的聚类中心
            prev_centers = centers
        else:
            status = 'normal'

        results.append((i, timestamp, value, status, centers, weights, distance))

    return results

# 示例数据
data = [
    (datetime(2023, 1, 1, 0, 0, i), val) for i, val in enumerate([
        10, 10, 10, 10, 10,  # 稳定段
        15, 20, 25, 30, 35,  # 逐渐上升
        40, 45, 50, 55, 60,  # 逐渐上升
        60, 60, 60, 60, 60,  # 稳定高值段
        30, 30, 30, 30, 30,  # 突然下降
    ])
]

# 参数设置
time_window = timedelta(seconds=4)  # 时间窗口大小，例如5秒
center_change_threshold = 0.3       # 聚类中心单次变化的阈值比例
cumulative_change_threshold = 0.5   # 聚类中心累计变化的阈值比例

# 运行检测
results = detect_changes_with_kmeans(data, time_window, center_change_threshold, cumulative_change_threshold)

# 输出结果
for idx, timestamp, value, status, centers, weights, distance in results:
    print(f"索引: {idx}, 时间: {timestamp}, 值: {value}, 状态: {status}, 聚类中心: {centers}, 权重: {weights}, 距离: {distance}")