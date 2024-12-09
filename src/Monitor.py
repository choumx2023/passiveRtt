import sys
sys.path.append('/Users/choumingxi/Documents/GitHub/newrtt/src')
import ipaddress
from collections import defaultdict, Counter
import time
import statistics
from random import randint
import logging
import os
import pickle
import math
import time
import seaborn
import math
from collections import deque
from ipaddress import IPv4Address, IPv6Address
import copy
import matplotlib.pyplot as plt
import numpy as np
from statistics import mean, stdev
# 这个class是基于Welford算法的，用于计算均值和方差
# 分类不同协议的，只在同类协议内比较
# 如果超过以当前窗口内的平均值为中心的一定范围后要报告，timestamp，rtt，delta给上层
from datetime import datetime, timedelta
import numpy as np
import math
def max_rank_distance(n, count):
    '''
    返回最大的秩距离
    params:
        n: 总的数据点
        count: 聚类的数据点
    '''
    if count == 0:
        return 0
    a = count//2
    b = count - a
    if count < 2:
        return 0
    r = 1.0 * n*(n+1)*(2*n+1)/6 + 1.0*b*(b+1)*(2*b+1)/6 - 1.0*(n-a)*(n-a+1)*(2*(n-a)+1)/6
    s = 1.0 * n*(n+1)/2 - (n-a-b)*(n-a+b+1)/2
    total = a+ b
    var = r/total - (s/total)**2
    return var

def kmeans_1d_with_tolerance(values, max_iter=10, outlier_threshold=2.0, tol=1e-5, verbose=False):
    """
    一维 K-means 聚类算法，第一轮迭代后剔除离群点，不参与中心计算，但保留标签。
    
    参数：
    - values: 一维数据列表
    - max_iter: 最大迭代次数
    - outlier_threshold: 离群点阈值，距离簇中心的倍数（默认 2.0）
    - tol: 判断收敛的容差（默认 1e-4）
    - verbose: 是否打印每次迭代的日志（默认 False）
    
    返回：
    - 'centers': 聚类中心列表 [center 0, center 1]
    - labels: 每个数据点的聚类标签列表 [label 0, label 1, ...]
    """
    if not values:
        return [], []
    if len(values) < 2:
        return values, [0] * len(values)
    vs = [i for i in values]
    # 初始化聚类中心
    sorted_values = sorted(values)
    centers = [sorted_values[0], sorted_values[(1 + 2 * len(values)) // 3] ]  # 最小值和第二大值
    labels = [-1] * len(values)  # 初始化标签
    labels = [0] * len(values)  # 初始化标签

    for iteration in range(max_iter):
        # 创建两个空的簇
        clusters = [[] for _ in range(len(centers))]
        labels_temp = []

        # 将每个点分配到最近的聚类中心
        for i, v in enumerate(values):
            distances = [abs(v - c) for c in centers]
            min_idx = distances.index(min(distances))
            labels_temp.append(min_idx)
            clusters[min_idx].append(v)

        # 在第一轮迭代后检测离群点
        if iteration in [1, 2]:
            non_outlier_values = []
            for i, cluster in enumerate(clusters):
                if cluster:
                    # 计算簇中心和平均距离
                    center = centers[i]
                    distances = [abs(v - center) for v in cluster]
                    avg_distance = sum(distances) / len(cluster)

                    for v in cluster:
                        if abs(v - center) > outlier_threshold * avg_distance or abs(v - center) < 1 / outlier_threshold * avg_distance:
                            continue  # 跳过离群点
                        else:
                            non_outlier_values.append(v)

            # 更新非离群点作为后续数据
            values = non_outlier_values

        # 计算新的聚类中心
        new_centers = []
        for i, cluster in enumerate(clusters):
            if cluster:
                new_centers.append(sum(cluster) / len(cluster))  # 使用簇中点的均值
            else:
                # 如果簇为空，保留原中心
                new_centers.append(centers[i])
        
        # 打印日志
        if verbose:
            print(f"Iteration {iteration + 1}:")
            print(f"  Centers: {centers}")
            print(f"  Clusters: {clusters}")
        
        # 检查是否收敛（允许容差）
        if all(abs(new_centers[i] - centers[i]) < tol for i in range(len(centers))):
            if verbose:
                print("Converged!")
            break
        
        centers = new_centers
        labels = labels_temp
    labels = [0] * len(vs)
    for i, j in enumerate(vs):
        distances = [abs(j - c) for c in centers]
        min_idx = distances.index(min(distances))
        labels[i] = min_idx
        
    print(centers, labels, vs)
    return centers, labels
def calculate_within_group_distance(values, centers, labels):
    '''
    计算聚类内的rtt的距离
    params:
        values: list of values
        'centers': list of centers
        labels: list of labels
    return :
        [center0_distance, center1_distance]
    '''
    total_distance = [0, 0]
    counts = [0, 0]
    for v, l in zip(values, labels):
        total_distance[l] += abs(v - centers[l])**2
        counts[l] += 1
    return [math.sqrt(total_distance[l] / (counts[l] if counts[l] != 0 else 1))  for l in range(2)]


def check_status(values, centers, labels):
    # 先查看偏心情况
    # 查看平均秩和整体均值的差别
    # 如果偏差很大，说明可能存在step increase或者step decrease 或者gradual increase或者gradual decrease
    # 如果聚类中心偏差比较大 说明step
    # 如果聚类中心偏差比较小，说明gradual
    
    # 如果偏差不是特别大，说明可能存在normal或者turbulence，或者凸凹
    # 如果秩的方差都是比较中等，而且方差接近，均值差的比较小，说明是normal
    # 如果方差比较中等，但是均值差的比较大，说明是turbulence
    # 如果一个方差比较大 另一个比较小，说明是凸凹
    
    pass
def calculate_rank_sum(values, labels):
    '''
    params:
        values: list of values
        labels: list
    return:
        [
            result:
                0 : 第一个聚类的平均秩
                1 : 第二个聚类的平均秩
                2 : 总体平均秩
                
                [ average_rank_sum_0, average_rank_sum_1, overall_average_rank_sum ] 
            rank_distance:
                0 : 第一个聚类的秩的方差
                1 : 第二个聚类的秩的方差
                2 : 第一个聚类的最小秩方差
                3 : 第二个聚类的最小秩方差
                4 : 第一个聚类的最大秩方差
                5 : 第二个聚类的最大秩方差
                [ rank_variance_0, rank_variance_1 ]
                
            e:
                0 : 第一个聚类到边界的理论最小秩距离
                1 : 第二个聚类到边界的理论最小秩距离
                2 : 第一个聚类到边界的最小距离
                3 : 第二个聚类到边界的最小距离
                
                [ min_rank_0, min_rank_1, min_distance_0, min_distance_1]
        ]
    对于min rank distance实际上是说明当前聚类中心距离最小的秩0和最大秩n-1的距离
    比如15个数据，序号为0-14，如果两个聚类的大小为4和11，那么第一个聚类的最小的距离是1.5，第二个聚类的最小距离是5
    也就是说第二个聚类有11个数据，最小的秩是0+1+2+3 = 6，平均是1.5，最大是14+13+12+11 = 12.5他们分别到两个边界（0，14）都是1.5
     
    '''
    rank_distance = [0, 0]
    rank_sum = [0, 0]
    count = [0, 0]
    rank = 0
    e = []
    lens = len(labels)
    for l in labels:
        rank_sum[l] += rank
        count[l] += 1
        rank_distance[l] += rank ** 2
        rank += 1
    # D(X) = 1/n Σ(Xi - X)^2 = E(X^2) - E(X)^2
    # Min(D(X)) = 1/2 * (n**2 - 1) / 12
    max_variance1 = max_rank_distance(lens, count[0])
    max_variance2 = max_rank_distance(lens, count[1])
    rank_distance = [
        (rank_distance[l] / count[l] - (rank_sum[l] / count[l]) ** 2) if count[l] != 0 else 0
        for l in range(2)
    ]
    rank_distance.append( (count[0]**2 - 1) / 12)
    rank_distance.append( (count[1]**2 - 1) / 12)
    rank_distance.append(max_variance1)
    rank_distance.append(max_variance2)
    avg_rank = [1.0 * rank_sum[l] / (count[l] if count[l] != 0 else 1) for l in range(2)]
    e =  [ (count[0] - 1) / 2, (-1 + count[1]) / 2, min (avg_rank[0], lens - 1 - avg_rank[0]), min(avg_rank[1], lens - 1 - avg_rank[1])]
    avg_rank.append((len(labels) - 1) / 2)
    return avg_rank, rank_distance, e  


class Analyser:
    
    def __init__(self, type, time_window = 60):
        self.window = deque()
        self.results = []
        self.time_window = time_window
        self.maximum_time_window = 60
        self.conclusion = []
        self.estimation = []
        self.type = type
    
    def detect_changes_with_kmeans(self, rtt_sample, timestamp):
        """
        使用 K-means 聚类方法检测时序数据中的突变和逐渐上升变化。

        参数：
        - time_window: 时间窗口大小，timedelta 对象
        - center_change_threshold: 聚类中心单次变化的阈值比例
        - cumulative_change_threshold: 聚类中心累计变化的阈值比例

        返回：
        - results: 包含每个数据点的状态列表，元素格式为 (index, timestamp, value, status)
          - status: 'normal', 'sudden_change', 'gradual_increase'
          - center = [cluster0, cluster1, mean_rtt, current_rtt]
        """
        centers, label, weights, distance, rank_sum, rank_distance, e = [None] * 7
        status = 'unknown'
        self.window.append((timestamp, rtt_sample))
        
        # 移除过时数据
        while self.window and (timestamp - self.window[0][0]) > self.time_window:
            if len(self.window) <= 10 and timestamp - self.window[0][0] < self.maximum_time_window:
                break
            self.window.popleft()
        # 如果数据点不足，直接返回
        if len(self.window) < 2:
            return {
                'rtt': rtt_sample,
                'timestamp': timestamp,
                'centers': [rtt_sample, 0, rtt_sample, rtt_sample],
                'label': 0,
                'labels': [0],
                'weights': [1, 0],
                'distance': [0.0] * 6,
                'rank_sum': [0.0] * 3,
                'rank_distance': [0, 0, 0, 0, 0, 0], 
                'e': [0, 0, 0, 0],
                'status': 'insufficient_data',
                'result_distance': ('small', 'small'),
                'elipse': 'normal'
            }
            
        # 2-mean聚类
        values_in_window = [val for _, val in self.window]
        centers, labels = kmeans_1d_with_tolerance(values_in_window)
        len_values = len(values_in_window)
        distance = calculate_within_group_distance(values_in_window, centers, labels)
        rank_sum, rank_distance, e  = calculate_rank_sum(centers, labels)
        min1, min2, max1, max2 = rank_distance[2:]
        len1, len2 = [len([l for l in labels if l == i]) for i in range(2)]
        actual = rank_distance[:2]
        def classify_point(actual, min1, max1, min2, max2, len1, len2):
    # 定义分位点
            total = len1 + len2
            interval1 = total / len1
            interval2 = total / len2
            q0 = [min1 * (interval1) ** 2 * 0.7 , min1 * 0.1 + max1 * 0.9]
            q1 = [min2 * (interval2) ** 2 * 0.7, min2 * 0.1 + max2 * 0.9]
            # 结果分类
            def assign_class(value, q_val, max_val, len_values):
                if len_values == 1:
                    return "small"
                
                if value <= q_val:
                    return "small"
                elif q_val < value <= max_val:
                    return "mid"
                else:
                    return "big"
            
            # 计算分类
            class1 = assign_class(actual[0], q0[0], q0[1], len1)
            class2 = assign_class(actual[1], q1[0], q1[1], len2)
            
            return class1, class2
        # 复制为small, mid, big
        result_distance = classify_point(actual, min1, max1, min2, max2, len1, len2)
        
        # 离心率，不偏，小聚类偏，小聚类和大聚类都偏
        
        weights = [
            len([l for l in labels if l == 0]) / len(labels),
            len([l for l in labels if l == 1]) / len(labels)
        ]
        label = labels[-1]
        total_distance = abs(weights[0] * (rank_sum[0] - len(labels)/2 - 0.5) )+ abs(weights[1] * (rank_sum[1] - len(labels)/2 - 0.5) )
        total_distance *= len_values
        min_weight = min(weights)
        max_weight = max(weights)
        if min(weights) < 0.2 :
            elipse = 'saltation'
        elif total_distance <= (len_values * min_weight) * (int(len_values * min_weight + 1.1) // 2) * 2.4:
            elipse = 'normal'
        elif total_distance <= (len_values * max_weight) * (int(len_values * min_weight + 1.1) // 2) * 2 :
            elipse = 'medium'
        else:
            elipse = 'big'
        centers.append(float(np.mean(values_in_window))) # 所有样本的平均值
        centers.append(centers[labels[-1]]) # 当前样本所属样本的聚类中心
        
        result_e = [ (rank_sum[0] * weights[0] + rank_sum[1] * weights[1]), rank_sum[2]]
        # 已有的标准
        # 如果数据点足够多，进行状态判断
        if len(self.window) >= 10:
            pass
        # 如果数据点不足，可以看一下分布情况
        else:
            # 数据点不足，无法进行聚类
            status = 'insufficient_data'
        return {
            'rtt': rtt_sample,
            'timestamp': timestamp,
            'centers': centers, 
            'weights':weights, #zb
            'label': label,
            'labels': labels,
            'weights': weights, #zb
            'distance': distance,
            'rank_sum': rank_sum,
            'rank_distance': rank_distance,
            'e': e,
            'status': status,
            'result_distance': result_distance, # zb
            'elipse': elipse #zb
        }

    def add(self, rtt, timestamp):
        conclusion = self.detect_changes_with_kmeans(rtt, timestamp)
        self.conclusion.append(conclusion)
        e = conclusion['e']
        label = conclusion['label']
        centers = conclusion['centers']
        rank_sum = conclusion['rank_sum']
        weights = conclusion['weights']
        # 缺少数据
        if conclusion['status'] == 'insufficient_data':
            self.estimation.append({'con': 'insufficient_data', 'rtt': rtt, 'timestamp': timestamp, 'rtt_estimation': centers[0] * weights[0] + centers[1] * weights[1], 'label': label, 'centers': centers, 'weights':weights})
            return
        elif conclusion['elipse'] == 'saltation':
            if weights[label] > 1 - 0.2:
                self.estimation.append({'con': 'normal', 'rtt': rtt, 'timestamp': timestamp, 'rtt_estimation': centers[label], 'label': label, 'centers': centers, 'weights':weights})
            else:
                self.estimation.append({'con': 'saltation', 'rtt': rtt, 'timestamp': timestamp, 'rtt_estimation': centers[1 - label], 'label': label, 'centers': centers, 'weights':weights})
            return
        
        # 无偏心
        elif conclusion['elipse'] == 'normal':
            # 如果都是mid，那么就是normal
            if conclusion['result_distance'] == ('mid', 'mid'):
                self.estimation.append({'con': 'normal', 'rtt': rtt, 'timestamp': timestamp, 'rtt_estimation': centers[label], 'label': label, 'centers': centers, 'weights':weights})
                
            elif (conclusion['result_distance'] == ('small', 'mid') and weights[0] >= weights[1]) or (conclusion['result_distance'] == ('mid', 'small') and weights[1] >= weights[0]) or conclusion['result_distance'] == ('small', 'small'):
                self.estimation.append({'con': 'step_change', 'rtt': rtt, 'timestamp': timestamp, 'rtt_estimation': centers[label], 'label': label, 'centers': centers, 'weights':weights})
                
            else:
                self.estimation.append({'con': 'turbulent_normal', 'rtt': rtt, 'timestamp': timestamp, 'rtt_estimation': centers[label], 'label': label, 'centers': centers, 'weights':weights})
        # 有小偏心
        elif conclusion['elipse'] == 'medium':
            
            if conclusion['result_distance'] == ('small', 'small'):
                self.estimation.append({'con': 'normal', 'rtt': rtt, 'timestamp': timestamp, 'rtt_estimation': centers[label], 'label': label, 'centers': centers, 'weights':weights})
            elif (conclusion['result_distance'] == ('small', 'mid') and weights[0] >= weights[1]) or (conclusion['result_distance'] == ('mid', 'small') and weights[1] >= weights[0]):
                self.estimation.append({'con': 'step_change', 'rtt': rtt, 'timestamp': timestamp, 'rtt_estimation': centers[label], 'label': label, 'centers': centers, 'weights':weights})
                
            else:
                self.estimation.append({'con': 'turbulent_mid', 'rtt': rtt, 'timestamp': timestamp, 'rtt_estimation': centers[label], 'label': label, 'centers': centers, 'weights':weights})
                
        # 有大偏心
        elif conclusion['elipse'] == 'big':
            
            if rtt > centers[1 - label] + conclusion['distance'][1 - label] and abs(centers[1 - label] - centers[label]) < 20:
                self.estimation.append({'con': 'gradual_increase', 'rtt': rtt, 'timestamp': timestamp, 'rtt_estimation': centers[label], 'label': label, 'centers': centers, 'weights':weights})
            elif rtt < centers[1 - label] - conclusion['distance'][1 - label] and abs(centers[label] - centers[1 - label]) < 20:
                self.estimation.append({'con': 'gradual_decrease', 'rtt': rtt, 'timestamp': timestamp, 'rtt_estimation': centers[label], 'label': label, 'centers': centers, 'weights':weights})
            
            
            elif rtt < centers[label] + conclusion['distance'][label] and rtt > centers[label] - conclusion['distance'][label] and centers[label] - centers[1 -label] > 20:
                self.estimation.append({'con': 'sudden_increase', 'rtt': rtt, 'timestamp': timestamp, 'rtt_estimation': centers[label], 'label': label, 'centers': centers, 'weights':weights})
            elif rtt < centers[label] + conclusion['distance'][label] and rtt > centers[label] - conclusion['distance'][label] and centers[1 - label] - centers[label] > 20:
                self.estimation.append({'con': 'sudden_decrease', 'rtt': rtt, 'timestamp': timestamp, 'rtt_estimation': centers[label], 'label': label, 'centers': centers, 'weights':weights})
                
                
            else:
                self.estimation.append({'con': 'turbulent_big', 'rtt': rtt, 'timestamp': timestamp, 'rtt_estimation': centers[label], 'label': label, 'centers': centers, 'weights':weights})
            return 
    
    def deliver(self):
        new_estimation = self.estimation[-1]
        upstream_data = {
            'type': self.type,
            'rtt': new_estimation['rtt'],
            'timestamp': new_estimation['timestamp'],
            'rtt_estimation': new_estimation['rtt_estimation'],
            'centers': new_estimation['centers'],
            'con': new_estimation['con']
        }
        return upstream_data
    def __print__(self):
        for v, l in zip(self.estimation, self.conclusion):
            print(v,'\n', l)
            print('-------------------')
   

## 00成功 01失败
class WelfordVariance:
    def __init__(self, time_window: int = 1200, max_count: int = 1200, initial_limit: int = 6):
        self.time_window = time_window  # 数据点有效的时间窗口
        self.max_count = max_count      # 数据点的最大数量
        self.initial_limit = initial_limit  # 初始数据点的数量，这些点不计入统计
        self.data = deque()  # 存储所有有效数据点
        self.initial_data = deque()  # 专门存储初始数据点
        self.count = 0  # 用于统计的数据点数量
        self.mean = 0   # 当前的均值
        self.M2 = 0     # 用于计算方差的中间变量
        self.recorded_count = 0  # 已记录的数据点数量
        
    #  60 50 50 70 200
    def update(self, x: float, timestamp: float):
        """更新统计数据，包括添加新数据点。"""
        # 判断是否仍在收集初始数据点
        if self.recorded_count < self.initial_limit:
            self.initial_data.append((x, timestamp))
            self.recorded_count += 1
            if len(self.initial_data) == self.initial_limit:
                # 初始数据点收集完毕，开始计算均值
                self.init_mean = sum(x for x, _ in self.initial_data) / self.initial_limit
                self.init_variance = sum((x - self.init_mean) ** 2 for x, _ in self.initial_data) / (self.initial_limit - 1)
        elif (self.recorded_count <= 2 * self.initial_limit and abs(x - self.init_mean) < min(20, max(15 , 0.95 * math.sqrt(self.init_variance), math.sqrt(self.init_variance)))) or self.recorded_count > 2 * self.initial_limit:
            # 更新统计计数和数据
            self.data.append((x, timestamp))
            self.count += 1
            self.recorded_count += 1
            delta = x - self.mean
            self.mean += delta / self.count
            delta2 = x - self.mean
            self.M2 += delta * delta2
            
            
            
            
    def variance(self):
        """计算并返回当前的方差。"""
        if self.count < 2:
            return float('nan')  # 数据点不足以计算方差
        return self.M2 / (self.count - 1)
    def get_mean(self):
        """返回当前的均值。"""
        return self.mean
    def remove_outdated(self, current_time: float):
        """移除过时的数据点，根据时间窗口判断。"""
        while self.data and (current_time - self.data[0][1]) > self.time_window:
            self.remove(self.data.popleft()[0])

    def remove(self, x: float):
        """移除指定的数据点，并更新统计。"""
        if self.count <= 1:
            self.reset()
        else:
            self.count -= 1
            delta = x - self.mean
            self.mean -= delta / self.count
            delta2 = x - self.mean
            self.M2 -= delta * delta2

    def reset(self):
        """重置所有统计信息。"""
        self.count = 0
        self.mean = 0
        self.M2 = 0
        self.data.clear()
        self.initial_data.clear()

    def str_variance(self):
        """返回当前统计信息的字符串表示。"""
        return f'Count: {self.count}, Variance: {self.variance()}, Mean: {self.mean}'

    def check_anomalies(self, newrtt: float, timestamp: float):
        """检查新数据点是否异常，并移除过时数据。"""
        self.remove_outdated(timestamp)
        if self.count >= 6:
            adjusted_variance = self.variance()
            if adjusted_variance is not float('nan'):
                threshold = min(20, max(10, 0.95 * math.sqrt(adjusted_variance)))
                if newrtt - self.mean > threshold:
                    return True
        self.update(newrtt, timestamp)
        return False
def default_state():
    return {
        'count': 0,
        'timestamps': []
    }
def default_rtt_state():
    return {
        'count': {},
        'rtt_values': []
    }
class CompressedIPNode:
    '''
    This class represents a node in a compressed trie for storing IP addresses. It supports both IPv4 and IPv6 addresses.
    It contains methods for recording RTT values and network activity, as well as aggregating statistics and detecting anomalies.

    attributes:
        network: The IP address network range of the node
        subnets: The subnets that were merged to create this node
        children: The child nodes of this node
        parent: The parent node of this node
        alerts: A list of alerts generated by this node
        logger: The logger object
        contain_ip_number: The number of IP addresses contained in the network
        contain_rtt_ip_number: The number of IP addresses contained in the network with RTT values
        stats: A dictionary of network activity statistics
        rtt_records: A dictionary of RTT records
        all_rtt_records: A list of all RTT records
        anomalous_rtts_records: A dictionary of anomalous RTT records
        rtt_stats: A dictionary of RTT statistics
        rtt_WelfordVariance: A WelfordVariance object for calculating RTT variance 
    '''
    def __init__(self, network : ipaddress.IPv4Address|ipaddress.IPv6Address, logger : str=None):
        '''

        params:
            network: IP地址的网络范围
            logger: 日志记录器
        初始化一个新的IP节点。
        '''
        self.network = network # 记录当前节点的网络范围
        self.subnets = []  # 记录由哪些子网合并而来
        self.children = {} # 存储子节点 (str, CompressedIPNode)
        self.parent = None  # 记录父节点
        self.parent : CompressedIPNode | None
        self.alerts = []
        self.logger = logger
        self.contain_ip_number = 0
        self.contain_rtt_ip_number = 0
        self.stats = defaultdict(default_state)
        
        # 以下是对延迟估计的统计数据
        self.delay_estimation = {
            'ICMP' : [],
            'NTP' : [],
            'DNS' : [],
            'TCP-Handshake' : [],
            'TCP-Normal' : [],
            'TCP-Timestamp' : [],
            'link_delay' : [],
            'TCP_Apllication_delay' : [],
            'DNS_iteration_delay' : []
        }
        
        
        # 以下是用于检测RTT的统计数据
        self.rtt_records = defaultdict(list) # 正常的rtt记录，key是(protocol, pattern)，value是rtt列表
        self.all_rtt_records = [] # 所有正常的rtt记录，不区分协议和模式
        self.all_delta_records = [] # 所有正常的rtt delta记录，不区分协议和模式
        
        # 以下是用于检测RTT的异常数据
        self.anomalous_rtts_records = defaultdict(list) # 异常的rtt记录，key是(protocol, pattern)，value是rtt列表
        self.subnets_anomalous_rtts_records = defaultdict(list) # 异常的rtt记录，key是(protocol, pattern)，value是rtt列表
        
        # 以下是用于检测RTT变化的异常数据
        self.anomalous_delta_rtts_records = defaultdict(list) # 异常的rtt记录，key是(protocol, pattern)，value是rtt, delta列表
        self.subnets_delta_anomalous_rtts_records = defaultdict(list) # 异常的rtt记录，key是(protocol, pattern)，value是rtt,delta列表
        
        self.rtt_stats = {
            'min_rtt': float('inf'),
            'max_rtt': float('-inf'), 
        }
        
        self.accumulate_normal_stats = defaultdict(int)
        self.accumulate_rtt_stats = defaultdict(int)
        self.accumulate_delta_stats = defaultdict(int)
        
        self.rtt_WelfordVariance = WelfordVariance(time_window=1000, max_count=1000)
        # 以下是用于检测网络活动的统计数据
        
        self.dns_WelfordVariance = WelfordVariance(time_window=1000, max_count=1000)
        self.icmp_ntp_WelfordVariance = WelfordVariance(time_window=1000, max_count=1000)
        self.tcp_WelfordVariance = WelfordVariance(time_window=1000, max_count=1000)
        self.all_delta_records = []
        
        # 以下记录流量记录
        self.flows_record = [] # live_span, ports, throughput, valid_throughput
    def aggregate_stats(self):
        '''
        聚合来自所有子节点的stats数据。
        因为太多了就不聚合了

        '''
        return 
        """聚合来自所有子节点的统计数据。"""
        aggregated_stats = defaultdict(default_state)
        for child in self.children.values():
            for key, value in child.stats.items():
                aggregated_stats[key]['count'] += value['count']
                aggregated_stats[key]['timestamps'].extend(value['timestamps'])
        self.stats = aggregated_stats
    def is_rtt_anomalous(self,  rtt : float, timestamp : float):
        '''
        params:
            rtt: RTT值
            timestamp: 时间戳
        返回RTT是否异常。
        '''
        # TODO: 实现检测RTT是否异常的逻辑
        if self.rtt_WelfordVariance.check_anomalies(rtt, timestamp):
            return True
        return False
    def aggregate_rtt(self):
        '''
        params:
            None
        聚合来自所有子节点的RTT数据。顺带更新rtt_stats。
        '''
        # 聚合来自所有子节点的RTT数据，要求只聚合不大的子网掩码长度
        # ipv4的最大子网掩码长度是24，ipv6的最大子网掩码长度是96
        max_mask = 24 if self.network.version == 4 else 24 * 4
        aggregated_rtt = defaultdict(list)
        for child in self.children.values():
            self.rtt_stats['min_rtt'] = min(self.rtt_stats['min_rtt'], child.rtt_stats['min_rtt'])
            self.rtt_stats['max_rtt'] = max(self.rtt_stats['max_rtt'], child.rtt_stats['max_rtt'])
            if len(child.rtt_records) < 5:
                continue
            if self.network.prefixlen >= max_mask: # 只聚合到一定的子网掩码长度
                for key, values in child.rtt_records.items(): # 只收集有效的rtt记录
                    aggregated_rtt[key].extend(values)
        self.rtt_records = aggregated_rtt

    def record_rtt(self, protocol : str, pattern : str, rtt : float, timestamp : float,  check_anomalies=False):
        '''
        适合最小的ip地址
        params:
            protocol: 协议
            pattern: 模式
            rtt: RTT值
            timestamp: 时间戳
            check_anomalies: 是否检查异常
        记录RTT值。
        '''
        key = (protocol, pattern)
        value = (rtt, timestamp)
        if protocol == "DNS":
            welford_variance = self.dns_WelfordVariance
        elif protocol in ["ICMP", "NTP"]:
            welford_variance = self.icmp_ntp_WelfordVariance
        elif protocol == "TCP":
            welford_variance = self.tcp_WelfordVariance
        else:
            self.logger.warning(f"Unknown protocol: {protocol}")
            return
        #这一个是检测全部rtt的 即将淘汰===MARK
        if check_anomalies and self.is_rtt_anomalous(rtt, timestamp): # 检查RTT是否异常，需要设置check_anomalies=True
            current_mean_rtt = self.rtt_WelfordVariance.get_mean()
            delta = rtt - current_mean_rtt
            anormal_value = (rtt, timestamp, current_mean_rtt, delta)
            # 子网检查
            if self.check_anomal_in_subnets(key, anormal_value):
                self.subnets_anomalous_rtts_records[key].append(anormal_value)
            else:
                self.anomalous_rtts_records[key].append(anormal_value)
            if self.logger:
                self.logger.warning(f'Anomalous RTT detected: {protocol} - {rtt}ms at {timestamp}')
        else: # 如果RTT正常，则记录到正常的rtt记录中
            if check_anomalies == False:
                self.rtt_WelfordVariance.update(rtt, timestamp)
            self.logger.info(f'Recorded RTT: {protocol} - {rtt}ms at {timestamp}')
            self.rtt_records[key].append((rtt, timestamp))
            self.all_rtt_records.append((rtt, timestamp))
            if self.rtt_WelfordVariance.recorded_count >=2 * self.rtt_WelfordVariance.initial_limit:
                if rtt < self.rtt_stats['min_rtt'] and rtt > 0 :
                    self.rtt_stats['min_rtt'] = rtt
                if rtt > self.rtt_stats['max_rtt'] and rtt < 1e4:
                    self.rtt_stats['max_rtt'] = rtt
                
            if self.logger:
                self.logger.debug(f'Recorded RTT: {protocol} - {rtt}ms at {timestamp}')
            # 父母就不检查了，向上传递
            if self.parent and self.rtt_WelfordVariance.recorded_count >= self.rtt_WelfordVariance.initial_limit:
                self.parent.upstream_rtt(protocol, pattern, rtt, timestamp)
        
        # # 如果需要检查异常，则按照不同的协议进行检查
        # current_mean_rtt = welford_variance.get_mean()
        # delta = rtt - current_mean_rtt
        # if check_anomalies and welford_variance.check_anomalies(rtt, timestamp):
        #     anormal_value = (rtt, timestamp, current_mean_rtt, delta)
        #     if self.check_anomal_in_subnets(key, anormal_value):
        #         self.subnets_delta_anomalous_rtts_records[key].append(anormal_value)
        #     else:
        #         self.anomalous_delta_rtts_records[key].append(anormal_value)
        #     if self.logger:
        #         self.logger.warning(f'Anomalous RTT delta detected: {protocol} - {rtt}:{delta}ms at {timestamp}')
        # # 如果不检查异常，或者没有检查出异常，则只记录到正常的rtt记录中
        # else:
        #     welford_variance.update(rtt, timestamp)
        #     if self.logger:
        #         self.logger.info(f'Recorded RTT: {protocol} - {rtt}:{delta}ms at {timestamp}')
        #     self.rtt_records[key].append((rtt, timestamp))
        #     self.all_rtt_records.append((rtt, timestamp))
        #     if self.rtt_WelfordVariance.recorded_count >=2 * self.rtt_WelfordVariance.initial_limit:
        #         if rtt < self.rtt_stats['min_rtt'] and rtt > 0 :
        #             self.rtt_stats['min_rtt'] = rtt
        #         if rtt > self.rtt_stats['max_rtt'] and rtt < 1e4:
        #             self.rtt_stats['max_rtt'] = rtt
        #     if self.logger:
        #         self.logger.debug(f'Recorded RTT: {protocol} - {rtt}ms at {timestamp}')
        #     # 父母就不检查了，向上传递
        #     if self.parent and self.rtt_WelfordVariance.recorded_count >= self.rtt_WelfordVariance.initial_limit:
        #         self.parent.upstream_rtt(protocol, pattern, rtt, timestamp)
             
            
    def upstream_rtt(self, protocol : str, pattern : str, rtt : float, timestamp : float):
        '''
        This function records the upstream RTT values, it delivers the RTT values to the parent node.
        params:
            protocol: 协议
            pattern: 模式
            rtt: RTT值
            timestamp: 时间戳
        上游RTT。
        '''
        key = (protocol, pattern)
        self.logger.info(f'Recorded RTT: {protocol} - {rtt}ms at {timestamp}')
        max_mask = 24 if self.network.version == 4 else 24 * 4
        # 只记录到一定的子网掩码长度
        if self.network.prefixlen >= max_mask:
            self.rtt_records[key].append((rtt, timestamp))
            self.all_rtt_records.append((rtt, timestamp))
        self.rtt_WelfordVariance.update(rtt, timestamp)
        # 无论多大，都更新最大和最小值
        
        if self.rtt_WelfordVariance.recorded_count >= self.rtt_WelfordVariance.initial_limit:
            if rtt < self.rtt_stats['min_rtt'] and rtt > 0 :
                self.rtt_stats['min_rtt'] = rtt
            if rtt > self.rtt_stats['max_rtt'] and rtt < 1e4:
                self.rtt_stats['max_rtt'] = rtt
        if self.parent:
            self.parent.upstream_rtt(protocol, pattern, rtt, timestamp)
    
    def record_activity_recursive(self, protocol : str, action : str, count = 1, timestamp = None, check_anomalies=False):
        '''
        This function records activity recursively.

        params:
            protocol: 协议
            action: 动作
            count: 数量
            timestamp: 时间戳
            check_anomalies: 是否检查异常
        递归记录活动。
        '''
        
        # Only check anomalies if flag is True
        max_mask = 24 if self.network.version == 4 else 24 * 4
        key = (protocol, action)
        self.stats[key]['count'] += count
        if timestamp:
            self.stats[key]['timestamps'].append(timestamp)
        if check_anomalies:
            self.detect_protocols_anomalie(protocol)

        # Decide whether to check for anomalies only once at the initial call
        if self.parent and self.parent.network.prefixlen >= max_mask:
            self.parent.record_activity_recursive(protocol, action, count, timestamp, check_anomalies)
    def get_contain_ip_number(self):
        '''
        This function calculates the number of IP addresses contained in the network.
        params:
            None
        returns:
            contain_ip_number(int): The number of IP addresses contained in the network.
        '''
        return self.contain_ip_number
    def update_contain_ip_number(self):
        if self.network.prefixlen == 32 and self.network.version == 4:
            self.contain_ip_number = 1
            if len(self.rtt_records) > 5:
                self.contain_rtt_ip_number = 1
        elif self.network.prefixlen == 128 and self.network.version == 6:
            self.contain_ip_number = 1
            if len(self.rtt_records) > 5:
                self.contain_rtt_ip_number = 1
        else:
            self.contain_ip_number = sum([child.get_contain_ip_number() for child in self.children.values()])
            self.contain_rtt_ip_number = sum([child.contain_rtt_ip_number for child in self.children.values()])
    def detect_protocols_anomalie(self, protocol):
        '''
        This function detects anomalies in the network traffic.
        havent been implemented yet.
        
        
        params:
            protocol: The protocol to check for anomalies
        returns:
            None
        '''
        # 示例：检测DNS请求和响应的数量差异
        if protocol == "DNS":
            requests = self.stats[("DNS", "Query")]['count']
            responses = self.stats[("DNS", "Response")]['count']
            if abs(requests - responses) > 100:  # 假设差异阈值为100
                anomaly = f"Anomaly detected: DNS requests ({requests}) and responses ({responses}) difference exceeds threshold"
                self.alerts.append(anomaly)
                print(anomaly)  # 或者将异常信息发送到日志系统或警报系统
        
    def get_rtt_stats(self):
        '''
        This function returns the RTT statistics for the network.
        params:
            None
        returns:
            rtt_stats(dict): The RTT statistics for the network
        '''
        all_rtts = []
        for rtts in self.rtt_records.values():
            all_rtts.extend([rtt for rtt, _ in rtts])
        if not all_rtts:
            return None
        return {
            'min_rtt': self.rtt_stats['min_rtt'],
            'max_rtt': self.rtt_stats['max_rtt'],
            'all_rtts': all_rtts
        }

    def get_subnets_rtt_stats(self):
        '''
        This function returns the RTT statistics for the subnets.
        params:
            None
            
        have not been implemented yet.
        '''
        min_rtt, max_rtt, all_rtts = float('inf'), float('-inf'), []
        for child in self.children.values():
            stats = child.get_rtt_stats()
            if stats:
                min_rtt = min(min_rtt, stats['min_rtt'])
                max_rtt = max(max_rtt, stats['max_rtt'])
                all_rtts.extend(stats['all_rtts'])
        if min_rtt == float('inf') or max_rtt == float('-inf'):
            return None
        return {
            'min_rtt': min_rtt,
            'max_rtt': max_rtt,
            'all_rtts': all_rtts
        }
    
    def get_32bit_ips_rtts(self):
        """
        获取当前节点下所有 32 位子节点的 IP 地址的 RTT 记录。
        """
        rtts = []
        
        # 遍历当前节点的子节点，收集 32 位 IP 地址的 RTT 记录
        for child in self.children.values():
            if child.network.prefixlen == 32:
                # 提取 32 位子节点的 RTT 记录
                rtts.extend(child.anomalous_rtts_records.values())
            else:
                # 递归调用，处理更深层级的子节点
                rtts.extend(child.get_32bit_ips_rtts())
        
        return rtts

    def check_anomal_in_subnets(self, key, anormal_value) -> bool:
        '''
        key : (protocol, pattern)
        anormal_value: (rtt, timestamp, current_mean_rtt, delta)
        检查子网中的异常。
        
        Parameters:
            key: 键，指定协议和模式
            anormal_value: 异常值，包含 (rtt, timestamp, current_mean_rtt, delta)
            
        Returns:
            bool: 是否为子网级别的异常
        '''
        timestamp = anormal_value[1]
        parent = self.parent
        if parent.network.prefixlen < 24:
            return False
        if parent.parent is not None and parent.parent.network.prefixlen >=24:
            parent = parent.parent
         
        # 获取同一子网下所有IP的异常RTT记录，限定在时间窗内
        anomalous_rtts = [
            record for record in self.anomalous_rtts_records.get(key, [])
            if record[1] + 10 > timestamp
        ]
        subnets_anomalous_rtts = [
            record for record in self.subnets_anomalous_rtts_records.get(key, [])
            if record[1] + 10 > timestamp
        ]
        
        # 合并同一时间窗内的异常RTT
        relevent_anomalous_rtts = anomalous_rtts + subnets_anomalous_rtts
        delta = anormal_value[3]
        
        # 检查是否存在相关的异常RTT
        def check_relevant_anomalous_rtts(rtts, delta):
            accumulated_delta = [rtt[3] for rtt in rtts]
            
            # 计算均值和标准差，并使用切比雪夫不等式检查
            if len(accumulated_delta) < 2:
                return False
            mean = np.mean(accumulated_delta)
            sigma = np.std(accumulated_delta)
            k = np.sqrt(1 / 0.1)  # 设置90%的概率阈值
            
            lower_bound = mean - k * sigma
            upper_bound = mean + k * sigma
            
            # 如果delta在均值的k倍sigma范围内，则可能存在相关趋势
            return lower_bound <= delta <= upper_bound

        # 如果没有相关异常RTT，则认为是单个IP的问题，否则是子网级别的异常
        if not check_relevant_anomalous_rtts(relevent_anomalous_rtts, delta):
            return False  # 没有其他相关异常，可能是单个IP的问题
        return True

    def __rtt__(self, prefix=''):
        prefix1 = prefix + '  '
        def format_output(rtts):
            #return f'\n{prefix1}  '.join([f'{rtt}ms, {timestamps}' for rtt, timestamps in rtts])
            return f'\n{prefix1}  '.join([', '.join(map(str, rtts[i:i+8])) for i in range(0, len(rtts), 8)])

        if (self.network.prefixlen >= 25 and self.network.version == 4) or (self.network.prefixlen >= 97 and self.network.version == 6):
            if self.rtt_records == {}:
                return f'{prefix}RTT Datas : \n{prefix1}No RTT data'
            rtt_info = '\n'.join(
            f'{prefix1}{protocol}**{pattern} RTT  count = {len(rtts)}: \n  {prefix1}{format_output(rtts)}'
            for (protocol, pattern), rtts in self.rtt_records.items()
            )
            return f'{prefix}RTT Datas :\n{rtt_info}'
        else:
            if self.accumulate_rtt_stats == {}:
                return f'{prefix}RTT Datas : \n{prefix1}No RTT data'
            rtt_info = '\n'.join(
            f'{prefix1}{protocol}**{pattern} RTT count = {count}:  {prefix1}{format_output([])}'
            for (protocol, pattern), count in self.accumulate_rtt_stats.items()
            )
        return f'{prefix}RTT Datas :\n{rtt_info}'
    def __delta__(self, prefix=''):
        prefix1 = prefix + '  '
        def format_output(deltas):
            return f'\n{prefix1}  '.join([', '.join(map(str, deltas[i:i+8])) for i in range(0, len(deltas), 8)])
        if (self.network.prefixlen >= 25 and self.network.version == 4) or (self.network.prefixlen >= 97 and self.network.version == 6):
            # 如果没有异常的RTT Delta记录，则返回无异常数据
            if self.anomalous_delta_rtts_records == {} and self.subnets_delta_anomalous_rtts_records == {}:
                return f'{prefix}Delta Data: \n{prefix1}No delta data'
            subnet_delta_anomalous_rtts = '\n'.join(
                f'{prefix1}{protocol}**{pattern} RTT count = {len(rtts)}:\n  {prefix1}{format_output(rtts)}'
                for (protocol, pattern), rtts in self.subnets_delta_anomalous_rtts_records.items())
            delta_info = '\n'.join(
                f'{prefix1}{protocol}**{pattern} RTT count = {len(rtts)}:\n  {prefix1}{format_output(rtts)}'
                for (protocol, pattern), rtts in self.anomalous_delta_rtts_records.items())
            return f'{prefix}Delta Data: \n{prefix}  Subnets Delta Data: \n{subnet_delta_anomalous_rtts}\n{prefix}  Delta IP Data: \n{delta_info}'
    def __anormalies__(self, prefix=''):
        '''
        params:
            prefix: 前缀
        返回异常数据。
        '''
        prefix1 = prefix + '  '
        if self.anomalous_rtts_records == {} and self.subnets_anomalous_rtts_records == {}:
            return f'{prefix}Anomalies Data: \n{prefix1}No anomalies data'
        def format_output(rtts):
            return f'\n{prefix1}  '.join([', '.join(map(str, rtts[i:i+8])) for i in range(0, len(rtts), 8)])
        
        subnet_anomalous_rtts = '\n'.join(
            f'{prefix1}{protocol}**{pattern} RTT count = {len(rtts)}:\n  {prefix1}{format_output(rtts)}'
            for (protocol, pattern), rtts in self.subnets_anomalous_rtts_records.items())
        
        rtt_info = '\n'.join(
            f'{prefix1}{protocol}**{pattern} RTT count = {len(rtts)}:\n  {prefix1}{format_output(rtts)}'
            for (protocol, pattern), rtts in self.anomalous_rtts_records.items())
        return f'{prefix}Anomalies Data: \n{prefix}  Subnets Anomalies Data: \n{subnet_anomalous_rtts}\n{prefix}  Anomalies IP Data: \n{rtt_info}'
    
    
    def __stats__(self, prefix=''):
        prefix1 = prefix + '  '
        def format_timestamps(timestamps):
            # 将时间戳分段，每段最多包含5个时间戳
            return f'\n{prefix1}  '.join([', '.join(map(str, timestamps[i:i+10])) for i in range(0, len(timestamps), 10)])
        if (self.network.prefixlen >= 25 and self.network.version == 4) or (self.network.prefixlen >= 97 and self.network.version == 6):
            if self.stats == {}:
                return f'{prefix}Stats Data: \n{prefix1}No stats data'
            accumulated_stats = '\n'.join(
            f'{prefix1}{protocol}={action} count = {details["count"]} :'
            for (protocol, action), details in self.stats.items())
            stats_info = '\n'.join(
            f'{prefix1}{protocol}={action} count = {details["count"]} :\n {prefix1} {format_timestamps(details["timestamps"])}'
            for (protocol, action), details in self.stats.items())
            return f'{prefix}Stats Data: \n{accumulated_stats}\n{prefix}details : \n{stats_info}'
        else:
            if self.accumulate_normal_stats == {}:
                return f'{prefix}Stats Data: \n{prefix1}No stats data'
            stats_info = '\n'.join(
            f'{prefix1}{protocol}={action} count = {count} : {prefix1} {format_timestamps([])}'
            for (protocol, action), count in self.accumulate_normal_stats.items())
        return f'{prefix}Stats Data: \n{stats_info}'


    def __basic__(self, prefix=''):
        if prefix:
            prefix = '|' + '-' * (len(prefix) - 1)
        wol = self.rtt_WelfordVariance.str_variance()
        basic_info = f"{prefix}{self.network}, : IP Count={self.contain_ip_number, self.contain_rtt_ip_number}, RTT range: {self.rtt_stats['min_rtt']}ms - {self.rtt_stats['max_rtt']}ms, wolvalue = {wol}"
        return basic_info
    def __tcpflow__(self, prefix=''):
        prefix1 = prefix + '  '
        def format_output(flows):
            return f'\n{prefix1}  '.join([f'{flow} : {value}' for flow, value in flows.items()])
        if self.flows_record == []:
            return f'{prefix}Flows Data : \n{prefix1}No flows data'
        flow_info = '\n'.join(
        f'{prefix1}Flow {count + 1}: \n{prefix1}  {format_output(flow)}'
        for count, flow in enumerate(self.flows_record)
        )
        return f'{prefix}Flows Data count = {len(self.flows_record)} :\n{flow_info}'
class CompressedIPTrie:
    '''
    This class represents a compressed trie for storing IP addresses. It supports both IPv4 and IPv6 addresses.
    '''
    def __init__(self, ip_version=4, logger=None):
        '''
        This function initializes a new compressed trie for storing IP addresses.
        params:
            ip_version: The IP version (4 or 6)
            logger: The logger object
        return:
            None
        '''
        self.root = CompressedIPNode(network=ipaddress.ip_network("0.0.0.0/0" if ip_version == 4 else "::/0", ), logger=logger)
        self.logger = logger    
        self.ip_version = ip_version
    def add_ip(self, ip : str):
        '''
        This function adds an IP address to the trie. If the IP address is already in the trie, it will not be added again, otherwise it will be created and added.
        If it is a new node, then it will be merged with its parent node if possible.
        
        params:
            ip: The IP address to add
        returns:
            None
        '''
    # 确保IP版本匹配
        if ipaddress.ip_address(ip).version != self.ip_version:
            return
        new_ip = f'{ip}/32' if self.ip_version == 4 else f'{ip}/128'
        new_net = ipaddress.ip_network(new_ip, strict=False)
        if not self.find_node(ip):
            new_node = CompressedIPNode(network=new_net, logger=self.logger)
            new_node.update_contain_ip_number()
            self.insert_network(new_node)
            self._merge_network(new_node)

    def insert_network(self, new_net: CompressedIPNode):
        '''
        This function inserts a new network into the trie. It will recursively find the correct insertion point.
        params:
            new_net: The new IP node
        returns:
            None     
        '''
        node = self.root
        # 需要递归查找正确的插入点
        while True:
            placed = False
            for child in list(node.children.values()):
                if new_net.network.subnet_of(child.network):
                    node = child
                    placed = True
                    break
            if not placed:
                break
        new_net.parent = node
        node.children[new_net.network] = new_net
        node.update_contain_ip_number()
    
    def _merge_network(self, new_net : CompressedIPNode) -> None:
        '''
        This function merges networks if possible. It will recursively merge networks up to the root node.
        params:
            new_net: The new IP node
        returns:
            None
        
        '''
        # 合并网络应当考虑可能需要递归上溯到不只是直接父节点
        step = 2 if self.ip_version == 4 else 4
        merge_count = 2 if self.ip_version == 4 else 4
        max_subnet =  8 if self.ip_version == 4 else 32
        parent = new_net.parent
        while parent and new_net.network.prefixlen <= max_subnet:
            super_net = new_net.network.supernet(new_prefix=new_net.network.prefixlen - step)
            if super_net.prefixlen % step != 0:
                print('Invalid supernet.')
                break
            elif super_net == parent.network:
                break
            # 检查是否有足够的子节点可以合并
            max_subnet_prefix = super_net.prefixlen - step
            eligible_children = [child for child in parent.children.values() if child.network.subnet_of(super_net)]
            max_subnet_children = [child for child in parent.children.values() if child.network.subnet_of(super_net) and child.network.prefixlen == max_subnet_children]
            # TODO : 可以根据需要调整合并的条件 首先就是根据子网的rtt数量 这里需要一个聚类方法，参考指标就是各个子网的rtt_stats
            
            
            if len(max_subnet_children) >= merge_count:
                supernet_node = CompressedIPNode(network=super_net, logger=self.logger)
                parent.children[super_net] = supernet_node
                for child in eligible_children:
                    del parent.children[child.network]
                    supernet_node.children[child.network] = child
                    child.parent = supernet_node
                    supernet_node.subnets.append(child.network)
                supernet_node.aggregate_stats()
                supernet_node.aggregate_rtt()   
                supernet_node.update_contain_ip_number()
                supernet_node.parent = parent
                parent = supernet_node.parent
                new_net = supernet_node
            else:
                break

    def find_node(self, ip: str) -> CompressedIPNode:
        '''
        This function finds the node in the trie that contains the given IP address.
        params:
            ip: The IP address to find
        returns:
            node(CompressedIPNode): The node that contains the IP address
        '''
        node = self.root
        ip_obj = ipaddress.ip_address(ip)
        if ip_obj.version != self.ip_version:
            return None
        target = ipaddress.ip_network(f'{ip}/{32 if ip_obj.version == 4 else 128}', strict=False)

        while True:
            found = False
            for child in node.children.values():
                if child.network == target:
                    return child
                elif target.subnet_of(child.network):
                    node = child
                    found = True
                    break
            if not found:
                return None
    def record_activity(self, ip : str, activity_type : str, count=1, timestamp=None):
        '''
        This function records activity for a given IP address. It will find the node that contains the IP address and record the activity.
        If the IP address is not in the trie, it will be added first.
        params: 
            ip: The IP address
            activity_type: The type of activity
            count: The count of the activity
            timestamp: The timestamp of the activity
        returns:
            None
        '''
        trie = self.ipv4_trie if ipaddress.ip_address(ip).version == 4 else self.ipv6_trie
        trie : CompressedIPTrie
        node = trie.find_node(ip)
        if node == None:
            trie.add_ip(ip)
            node = trie.find_node(ip)
        if node:
            node.record_activity(activity_type, count, timestamp)
    def print_tree(self, node = None, indent = 0, file_path = 'tree.txt'):
        '''
        params:
            node: 节点
            indent: 缩进
            file_path: 文件路径
        打印树。
        '''
        if node is None:
            node = self.root

        indent_str = '    ' * indent
        with open(file_path, 'a') as f:
            f.write(f'{node.__basic__(indent_str)}\n')
            f.write(f"{node.__stats__(indent_str)}\n")
            f.write(f'{node.__tcpflow__(indent_str)}\n')
            f.write(f"{node.__rtt__(indent_str)}\n")
            f.write(f'{node.__anormalies__(indent_str)}\n')
        for child in node.children.values():
            self.print_tree(child, indent + 1, file_path)
    def collect_smallest_subnets(self):
        smallest_subnets = []
        self._collect_smallest_subnets_helper(self.root, smallest_subnets)
        return smallest_subnets

    def _collect_smallest_subnets_helper(self, node, smallest_subnets):
        # Define what is considered a "smallest subnet"
        if (self.ip_version == 4 and node.network.prefixlen == 32) or (self.ip_version == 6 and node.network.prefixlen == 128):
            smallest_subnets.append(node)
        for child in node.children.values():
            self._collect_smallest_subnets_helper(child, smallest_subnets)
    def waterfall_trees(self, node : CompressedIPNode):
        # 递归处理子节点
        if (node.network.prefixlen == 32 and node.network.version == 4) or (node.network.prefixlen == 128 and node.network.version == 6):
            node.contain_ip_number = 1
            total_length = 0
            for value_list in node.rtt_records.values():
                total_length += len(value_list)
            if total_length > 5:
                node.contain_rtt_ip_number = 1
            return
        node.contain_ip_number = 0
        node.contain_rtt_ip_number = 0
        is_bigger_network = (node.network.prefixlen <= 24 and node.network.version == 4) or (node.network.prefixlen <= 96 and node.network.version == 6)
        # 递归处理子节点
        for child in node.children.values():
            self.waterfall_trees(child)        
            # 更新节点的统计数据
            node.contain_ip_number += child.contain_ip_number
            node.contain_rtt_ip_number += child.contain_rtt_ip_number
            # 如果是较大的网络，不需要累加RTT数据
            if not is_bigger_network:
                for (protocol, pattern), rtts in child.rtt_records.items():
                    node.accumulate_rtt_stats[(protocol, pattern)] += len(rtts)
                for (protocol, action), details in child.stats.items():
                    node.accumulate_normal_stats[(protocol, action)] += details['count']
                for (protocol, pattern), rtts in child.anomalous_rtts_records.items():
                    node.accumulate_rtt_stats[('Anomalous '+protocol, pattern)] += len(rtts)
                    

    def analyse(self, node):
        packet_length_records = defaultdict(int)  # 用于存储前两位小数的统计结果

        # 封装网络条件的检查逻辑
        def is_valid_network(node):
            return (node.network.prefixlen == 32 and node.network.version == 4) or \
                (node.network.prefixlen == 128 and node.network.version == 6)

        def helper(node):
            # 只有在满足网络条件时才继续处理
            if is_valid_network(node):
                for flow in node.flows_record:
                    avg_packet_length = flow.get('average_packet_length', None)
                    avg_packet_length = avg_packet_length[0]
                    if avg_packet_length <0:
                        print(node.flows_record)
                        exit(1)
                    rounded_avg = round(avg_packet_length , 1)   # 保留前两位小数
                    packet_length_records[rounded_avg] += 1  # 计数

                # 递归处理子节点
            for child in node.children.values():
                helper(child)

        helper(node)  # 启动递归遍历
        return packet_length_records
class NetworkTrafficMonitor:
    def __init__(self, name = '', check_anomalies = 'True', logger = None):
        '''
        params:
            name: 名称
            check_anomalies: 是否检查异常
            logger: 日志记录器
        初始化网络流量监控器。
        '''
        self.ipv4_trie = CompressedIPTrie(ip_version=4, logger=logger)
        self.ipv6_trie = CompressedIPTrie(ip_version=6, logger=logger)
        self.timeslot = 0.2
        self.suffix = name
        self.check_anomalies = check_anomalies
    def add_ip_and_record_activity(self, ip, protocol, action, count=1, timestamp=None):
        '''
        params:
            ip: IP地址
            protocol: 协议
            action: 动作
            count: 数量
            timestamp: 时间戳
        添加IP地址并记录活动。
        '''
        # 确定使用IPv4还是IPv6的Trie
        trie = self.ipv4_trie if ipaddress.ip_address(ip).version == 4 else self.ipv6_trie
        
        # 尝试找到已存在的节点
        node = trie.find_node(ip)
        
        # 如果节点不存在，先添加IP
        if not node:
            trie.add_ip(ip)
            node = trie.find_node(ip)  # 重新获取新添加的节点
        
        # 记录活动，更新节点及其所有父节点
        if node:
            node.record_activity_recursive(protocol, action, count, timestamp)
    def add_flow_record(self, ip_pairs : list, flow_record : dict):
        '''
        params:
            flow_record: 流记录
        添加流记录。
        '''
        '''
        {
            'live_span': self.live_span,
            'throught_output': self.throught_output,
            'valid_throughput': self.valid_throughput,
            'max_length': self.max_length
        }
        '''
        
        forward_ip, backward_ip = ip_pairs[0], ip_pairs[1]
        reverse_data = copy.deepcopy(flow_record)
        reverse_data['live_span'] = flow_record['live_span']
        reverse_data['max_length'] = [flow_record['max_length'][1], flow_record['max_length'][0]]
        reverse_data['throught_output'] = [flow_record['throught_output'][1], flow_record['throught_output'][0]]
        reverse_data['valid_throughput'] = [flow_record['valid_throughput'][1], flow_record['valid_throughput'][0]]
        reverse_data['total_throughput'] = [flow_record['total_throughput'][1], flow_record['total_throughput'][0]]
        reverse_data['packet_count'] = [flow_record['packet_count'][1], flow_record['packet_count'][0]]
        reverse_data['average_packet_length'] = [flow_record['average_packet_length'][1], flow_record['average_packet_length'][0]]
        trie = self.ipv4_trie if ipaddress.ip_address(forward_ip).version == 4 else self.ipv6_trie
        
        node1 = trie.find_node(forward_ip)
        if not node1:
            trie.add_ip(forward_ip)
            node1 = trie.find_node(forward_ip)
        if node1:
            node1.flows_record.append(flow_record)
        node2 = trie.find_node(backward_ip)
        if not node2:
            trie.add_ip(backward_ip)
            node2 = trie.find_node(backward_ip)
        if node2:
            node2.flows_record.append(reverse_data)    
    def query_rtt(self, ip, protocol):
        '''
        params:
            ip: IP地址
            protocol: 协议
        查询RTT。
        '''
        trie = self.ipv4_trie if ipaddress.ip_address(ip).version == 4 else self.ipv6_trie
        node = trie.find_node(ip)
        if node and protocol in node.rtt_records:
            return statistics.mean(node.rtt_records[protocol]) if node.rtt_records[protocol] else None
        return None
    def query_activity(self, ip : IPv4Address | IPv6Address, protocol :str, action :str):
        '''
        params:
            ip: IP地址
            protocol: 协议
            action: 动作
        查询活动。
        '''
        trie = self.ipv4_trie if ipaddress.ip_address(ip).version == 4 else self.ipv6_trie
        node = trie.find_node(ip)
        if node:
            key = (protocol, action)
            total_count = sum(item[0] for item in node.stats[key])  # 汇总所有count
            timestamps = [item[1] for item in node.stats[key]]  # 获取所有时间戳
            return total_count, timestamps
        return 0, []
    def add_or_update_ip_with_rtt(self, ip : str, protocol : str, pattern : str, rtt : float, timestamp : float):
        '''
        params:
            ip: IP地址
            protocol: 协议
            pattern: 模式
            rtt: RTT值
            timestamp: 时间戳
        添加或更新IP地址和RTT。
        '''
        trie = self.ipv4_trie if ipaddress.ip_address(ip).version == 4 else self.ipv6_trie
        node = trie.find_node(ip)
        # 如果IP节点不存在，则添加它
        if not node:
            trie.add_ip(ip)
            node = trie.find_node(ip)  # 确保节点被添加后重新获取它
        # 现在记录RTT数据，假设节点现在肯定存在
        if node:
            # 如果self.check_anomalies为True，则检查异常
            node.record_rtt(protocol, pattern, rtt, timestamp, check_anomalies= self.check_anomalies)
        # 可选：检测异常情况
        if 0:
            node.check_rtt_anomalies()
    def detect_attack(self, ip, threshold = 1000):
        '''
        params:
            ip: IP地址
            threshold: 阈值
        检测攻击。
        '''
        trie = self.ipv4_trie if ipaddress.ip_address(ip).version == 4 else self.ipv6_trie
        node = trie.find_node(ip)
        if node:
            total_requests = sum(node.stats.values())
            return total_requests > threshold
        return False
    def print_trees(self):
        self.ipv4_trie.waterfall_trees(self.ipv4_trie.root)
        self.ipv6_trie.waterfall_trees(self.ipv6_trie.root)
        with open(f'{self.suffix}_tree.txt', 'w') as f:
            f.write("")  # Clear the contents of the file
        self.ipv4_trie.print_tree(file_path=f'{self.suffix}_tree.txt')
        #print("IPv4 Trie is saved to tree.txt")
        self.ipv6_trie.print_tree(file_path=f'{self.suffix}_tree.txt')
        #print("IPv6 Trie is saved to tree.txt")
        
    def analyze_traffic(self):
        v4_dict = self.ipv4_trie.analyse(self.ipv4_trie.root)
        v6_dict = self.ipv6_trie.analyse(self.ipv6_trie.root)

        def sort_and_plot(data: defaultdict):
            # 1. 对字典按键（average_packet_length）进行排序
            sorted_data = dict(sorted(data.items()))  # items() 返回字典的键值对

            # 2. 分别获取排序后的键和值
            x = list(sorted_data.keys())  # 排序后的 average_packet_length
            y = list(sorted_data.values())  # 对应的 count

            # 3. 绘制条形图（原有功能）
            plt.figure(figsize=(10, 6))
            plt.bar(x, y, color='blue')  # 使用条形图展示
            plt.title("Packet Length Frequency")
            plt.xlabel("Average Packet Length (rounded to 2 decimal places)")
            plt.ylabel("Count")
            plt.show()

            # 4. 计算 PDF
            total_count = sum(y)
            pdf = [count / total_count for count in y]

            # 5. 绘制 PDF 图
            plt.figure(figsize=(10, 6))
            plt.plot(x, pdf, color='green', marker='o')  # 绘制 PDF 曲线图
            plt.title("Probability Density Function (PDF)")
            plt.xlabel("Average Packet Length (rounded to 2 decimal places)")
            plt.ylabel("Probability")
            plt.grid(True)
            plt.show()

            # 6. 计算 CDF
            cdf = [sum(pdf[:i + 1]) for i in range(len(pdf))]

            # 7. 绘制 CDF 图
            plt.figure(figsize=(10, 6))
            plt.plot(x, cdf, color='r', marker='.')  # 绘制 CDF 曲线图
            plt.ylim(-0.1, 1.1)
            plt.xlim(0, 1600)
            plt.title("Cumulative Distribution Function (CDF)")
            plt.xlabel("Average Packet Length (rounded to 2 decimal places)")
            plt.ylabel("Cumulative Probability")
            plt.grid(True)
            plt.show()

        # 调用对 IPv4 和 IPv6 数据进行排序并绘制图表
        sort_and_plot(v4_dict)
        sort_and_plot(v6_dict)
            
    def save_state(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
    def merge_nodes(self, node_a : CompressedIPNode, node_b : CompressedIPNode):
        # 合并统计数据和RTT记录
        for key, stats in node_b.stats.items():
            if key in node_a.stats:
                node_a.stats[key]['count'] += stats['count']
                node_a.stats[key]['timestamps'].extend(stats['timestamps'])
            else:
                node_a.stats[key] = stats
        for key, rtts in node_b.rtt_records.items():
            if key in node_a.rtt_records:
                node_a.rtt_records[key].extend(rtts)
            else:
                node_a.rtt_records[key] = rtts
        if node_b.rtt_stats['min_rtt'] < node_a.rtt_stats['min_rtt']:
            node_a.rtt_stats['min_rtt'] = node_b.rtt_stats['min_rtt']
        if node_b.rtt_stats['max_rtt'] > node_a.rtt_stats['max_rtt']:
            node_a.rtt_stats['max_rtt'] = node_b.rtt_stats['max_rtt']
        # 递归合并子节点
        for subnet, child_node_b in node_b.children.items():
            if subnet in node_a.children:
                self.merge_nodes(node_a.children[subnet], child_node_b)
            else:
                node_a.children[subnet] = child_node_b
                child_node_b.parent = node_a

    def merge_smallest_network(self, node_parent, ip_version=4):
        prefix = 32 if ip_version == 4 else 128
        trie = self.ipv4_trie if ip_version == 4 else self.ipv6_trie

        for child in node_parent.children.values():
            if child.network.prefixlen == prefix:
                target_node = trie.find_node(str(child.network.network_address))
                if target_node is None:
                    trie.add_ip(str(child.network.network_address))
                    target_node = trie.find_node(str(child.network.network_address))
                self.merge_nodes(target_node, child)
            else:
                self.merge_smallest_network(child, ip_version)

    def merge_monitor(self, other_monitor : 'NetworkTrafficMonitor'):
        # 合并 IPv4 和 IPv6 Trie 的根节点
        self.merge_smallest_network(other_monitor.ipv4_trie.root, ip_version=4)
        self.merge_smallest_network(other_monitor.ipv6_trie.root, ip_version=6)

    @staticmethod
    def load_state(filename : str):
        with open(filename, 'rb') as f:
            return pickle.load(f)
    def clear_state(self):
        self.ipv4_trie = CompressedIPTrie(ip_version=4, logger=self.logger)
        self.ipv6_trie = CompressedIPTrie(ip_version=6, logger=self.logger)
    def __repr__(self) -> str:
        return (f'NetworkTrafficMonitor({self.ipv4_trie}, {self.ipv6_trie})')
    def __str__(self) -> str:
        return (f'NetworkTrafficMonitor({self.ipv4_trie}, {self.ipv6_trie})')

def generate_single_ip(base_net, prefix):
    '''
    params:
        base_net: 基础网络
        prefix: 前缀
    生成一个随机IP地址，基于指定的基础网络和子网掩码。
    '''
    """生成一个随机IP地址，基于指定的基础网络和子网掩码。"""
    network = ipaddress.ip_network(base_net)
    subnets = list(network.subnets(new_prefix=prefix))
    subnet = subnets[randint(0, len(subnets) - 1)]
    ip = str(subnet[randint(1, subnet.num_addresses - 2)])
    return ip
def setup_logging():
    '''
    params:
        None
    设置日志记录器。
    '''
    log_directory = "./logs"
    log_path = os.path.join(log_directory, f"network_monitor.log")

    if not os.path.exists(log_directory):
        os.makedirs(log_directory)

    logging.basicConfig(
        filename=log_path,
        filemode='a',
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    return logging.getLogger()

# 调用 setup_logging 函数以初始化日志设置
def read_pickle(filename):
    '''
    params:
        filename: 文件名
    读取pickle文件。
    '''
    with open (filename, 'rb') as f:
        data = pickle.load(f)
        return data
def main():
    logger = setup_logging()
    monitor = NetworkTrafficMonitor(logger = logger)
    
    monitor.add_ip_and_record_activity('192.168.1.200', 'DNS', 'Query', 1, time.time())
    monitor.add_or_update_ip_with_rtt('192.168.1.100', 'TCP', 'Timestamp',  300, time.time()) 
    monitor.add_ip_and_record_activity('192.168.1.201', 'DNS', 'Query', 1, time.time())
    monitor.add_or_update_ip_with_rtt('192.168.1.101', 'TCP', 'Timestamp',  300, time.time()) 
    monitor.add_ip_and_record_activity('192.168.1.202', 'DNS', 'Query', 1, time.time())
    monitor.add_or_update_ip_with_rtt('192.168.1.102', 'TCP', 'Timestamp',  300, time.time()) 
    monitor.add_ip_and_record_activity('192.168.1.203', 'DNS', 'Query', 1, time.time())
    monitor.add_or_update_ip_with_rtt('192.168.1.103', 'TCP', 'Timestamp',  300, time.time()) 
    monitor.add_ip_and_record_activity('192.168.1.204', 'DNS', 'Query', 1, time.time())
    monitor.add_or_update_ip_with_rtt('192.168.1.104', 'TCP', 'Timestamp',  300, time.time()) 
    monitor.add_ip_and_record_activity('192.168.1.205', 'DNS', 'Query', 1, time.time())
    monitor.add_or_update_ip_with_rtt('192.168.1.105', 'TCP', 'Timestamp',  300, time.time()) 
    monitor.add_or_update_ip_with_rtt('192.168.1.105', 'TCP', 'SYN',  700, time.time()) 
    monitor.add_ip_and_record_activity('192.168.1.206', 'DNS', 'Query', 1, time.time())
    monitor.add_or_update_ip_with_rtt('192.168.1.106', 'TCP', 'Timestamp',  300, time.time()) 
    monitor.add_ip_and_record_activity('192.168.1.207', 'DNS', 'Query', 1, time.time())
    monitor.add_or_update_ip_with_rtt('192.168.1.108', 'TCP', 'Timestamp',  300, time.time()) 
    monitor.add_ip_and_record_activity('192.168.1.210', 'DNS', 'Query', 1, time.time())
    monitor.add_or_update_ip_with_rtt('192.168.1.109', 'TCP', 'Timestamp',  300, time.time())
    monitor.add_ip_and_record_activity('192.168.1.230', 'DNS', 'Query', 1, time.time())
    monitor.add_ip_and_record_activity('192.168.1.230', 'DNS', 'Query', 1, time.time())
    monitor.add_or_update_ip_with_rtt('192.168.1.110', 'TCP', 'Timestamp',  300, time.time())
    monitor.add_ip_and_record_activity('192.168.1.120', 'TCP', 'forward',1, time.time())
    monitor.add_or_update_ip_with_rtt('192.168.1.120', 'TCP', 'Timestamp',  300, time.time())



    monitor.add_ip_and_record_activity('2001:db8::1', 'DNS', 'Query', 1, time.time())
    monitor.add_or_update_ip_with_rtt('2001:db8::2', 'TCP', 'Timestamp',  300, time.time())
    monitor.add_ip_and_record_activity('2001:db8::3', 'DNS', 'Query', 1, time.time())
    monitor.add_or_update_ip_with_rtt('2001:db8::4', 'TCP', 'Timestamp',  300, time.time())
    monitor.add_ip_and_record_activity('2001:db8::5', 'DNS', 'Query', 1, time.time())
    monitor.add_or_update_ip_with_rtt('2001:db8::6', 'TCP', 'Timestamp',  300, time.time())
    monitor.add_ip_and_record_activity('2001:db8::7', 'DNS', 'Query', 1, time.time())
    monitor.add_or_update_ip_with_rtt('2001:db8::8', 'TCP', 'Timestamp',  300, time.time())
    monitor.add_ip_and_record_activity('2001:db8::9', 'DNS', 'Query', 1, time.time())
    monitor.add_or_update_ip_with_rtt('2001:db8::10', 'TCP', 'Timestamp',  300, time.time())
    monitor.add_or_update_ip_with_rtt('2001:db8::10', 'TCP', 'SYN',  700, time.time())
    monitor.add_ip_and_record_activity('2001:db8::11', 'DNS', 'Query', 1, time.time())
    monitor.add_or_update_ip_with_rtt('2001:db8::12', 'TCP', 'Timestamp',  300, time.time())
    monitor.add_ip_and_record_activity('2001:db8::13', 'DNS', 'Query', 1, time.time())
    monitor.add_or_update_ip_with_rtt('2001:db8::14', 'TCP', 'Timestamp',  300, time.time())
    monitor.add_ip_and_record_activity('2001:db8::15', 'DNS', 'Query', 1, time.time())
    monitor.add_or_update_ip_with_rtt('2001:db8::16', 'TCP', 'Timestamp',  300, time.time())
    monitor.add_ip_and_record_activity('2001:db8::17', 'DNS', 'Query', 1, time.time())
    monitor.add_or_update_ip_with_rtt('2001:db8::18', 'TCP', 'Timestamp',  300, time.time())
    monitor.add_ip_and_record_activity('2001:db8::19', 'DNS', 'Query', 1, time.time())
    monitor.add_ip_and_record_activity('2001:db80:0000::20', 'TCP', 'forward', 2, time.time())
    # monitor.add_record('DNS', 'Query', num = 10, timestamps)
    # 打印Trie树以查看子网聚合情况
    monitor.print_trees()
    monitor.save_state('network_monitor.pkl')

def test():
    monitor = read_pickle('current_monitor.pkl')
    print(monitor)
    monitor.print_trees()
def analyser_test():
    analyesr = Analyser(type= 'TCP')
    data1 = [(28.199, 1669599832.325483), (29.334, 1669599832.327243), (37.837, 1669599832.367833), (27.378, 1669599832.370148), (40.122, 1669599832.370175), (27.74, 1669599832.400521), (34.214, 1669599832.404559), (42.005, 1669599832.409983),
        (27.42, 1669599832.428306), (26.87, 1669599832.832592), (27.782, 1669599832.864788), (28.104, 1669599833.021115), (33.982, 1669599833.449475), (28.179, 1669599833.474618), (28.817, 1669599833.474629), (28.711, 1669599833.503645),
        (29.224, 1669599833.50903), (29.432, 1669599833.50905), (27.861, 1669599833.518924), (27.414, 1669599833.521497), (27.293, 1669599833.531334), (29.573, 1669599833.53364), (27.253, 1669599833.533689), (27.687, 1669599833.537433),
        (38.653, 1669599833.547905), (26.091, 1669599833.550703), (27.912, 1669599833.552492), (28.522, 1669599833.562895), (29.496, 1669599833.567764), (27.981, 1669599833.567887), (38.539, 1669599833.572476), (27.501, 1669599833.590776),
        (41.675, 1669599833.592585), (40.019, 1669599833.592692), (27.927, 1669599833.596009), (28.025, 1669599833.596095), (28.29, 1669599833.809434), (28.506, 1669599833.809467), (28.533, 1669599833.809494), (30.788, 1669599833.811707),
        (32.41, 1669599833.813829), (70.117, 1669599833.815741), (36.495, 1669599833.817669), (40.016, 1669599833.821601), (27.962, 1669599833.836305), (28.147, 1669599833.836321), (28.149, 1669599833.836325), (26.586, 1669599833.836328),
        (29.752, 1669599833.838087), (30.587, 1669599833.840242), (27.118, 1669599833.866052), (27.32, 1669599833.866119), (27.384, 1669599833.866212), (27.416, 1669599833.866263), (27.46, 1669599833.868453), (30.479, 1669599833.873287),
        (27.387, 1669599833.878731), (28.117, 1669599833.89671), (32.262, 1669599833.89895), (34.989, 1669599833.901588), (27.872, 1669599833.909281), (35.863, 1669599833.909315), (28.252, 1669599833.90934), (36.249, 1669599833.917314),
        (39.881, 1669599833.920464), (28.462, 1669599833.925552), (28.253, 1669599833.927469), (25.737, 1669599833.927543), (28.692, 1669599833.938198), (40.811, 1669599833.950263), (26.677, 1669599833.977487), (28.289, 1669599833.995319),
        (34.902, 1669599834.017424), (28.188, 1669599834.055537), (27.682, 1669599834.087348), (38.622, 1669599834.126248), (27.931, 1669599834.154336), (32.452, 1669599834.206802), (36.687, 1669599834.823153), (28.479, 1669599837.221856),
        (28.565, 1669599837.221942), (28.41, 1669599852.407904), (31.347, 1669599857.513557), (28.929, 1669599857.542659)]
    data2 = [        (15.914, 1714302616.731294), (0.96, 1714302616.738443), (3.104, 1714302616.747924), (2.5, 1714302616.755249), (5.869, 1714302616.77491), (0.812, 1714302616.783157), (2.362, 1714302616.794052), (1.51, 1714302616.806456),
        (3.011, 1714302616.819512), (2.165, 1714302616.829932), (2.652, 1714302616.834147), (0.265, 1714302616.837367), (0.696, 1714302616.842665), (0.154, 1714302616.846526), (3.679, 1714302616.854469), (0.117, 1714302616.858056),
        (3.211, 1714302617.006853), (3.774, 1714302617.019782), (8.551, 1714302617.041471), (24.053, 1714302632.32768), (34.23, 1714302639.128214), (39.389, 1714302639.17871), (33.827, 1714302647.634252), (31.009, 1714302678.62056)]
    
    data3 = [
        (24.222, 1714302533.662023), (26.061, 1714302533.689027), (100.559, 1714302533.801121), (26.083, 1714302535.349473), (24.23, 1714302536.645992), (23.307, 1714302536.672641), (16.473, 1714302537.636312), (21.128, 1714302538.334429),
        (20.101, 1714302541.878189), (26.843, 1714302542.001027), (18.916, 1714302542.144326), (23.096, 1714302542.172398), (24.549, 1714302543.440087), (24.431, 1714302543.667026), (23.204, 1714302544.885248), (24.086, 1714302544.910915),
        (1.746, 1714302546.859467), (1.857, 1714302546.86302), (2.967, 1714302546.867688), (2.735, 1714302546.871239), (2.775, 1714302546.875161), (1.896, 1714302546.878842), (3.892, 1714302546.884276), (5.199, 1714302546.890007),
        (1.446, 1714302546.910816), (23.443, 1714302548.367455), (14.371, 1714302548.645769), (24.002, 1714302548.744355), (16.121, 1714302548.831743), (20.072, 1714302548.934866), (23.35, 1714302552.716028), (15.034, 1714302553.755999),
        (23.486, 1714302554.784831), (19.041, 1714302558.713061), (20.475, 1714302560.352693), (24.007, 1714302564.17473), (23.744, 1714302564.637982), (19.127, 1714302574.138234), (8.69, 1714302578.698438), (2.379, 1714302578.704133),
        (2.572, 1714302578.708627), (6.439, 1714302578.716768), (23.943, 1714302583.641806), (23.517, 1714302583.668674), (19.304, 1714302584.169755), (23.933, 1714302584.666077), (23.596, 1714302588.820846), (23.991, 1714302594.787409),
        (23.745, 1714302604.136483), (17.336, 1714302604.638784), (15.795, 1714302614.140831), (6.45, 1714302614.149769), (18.707, 1714302624.134597), (0.721, 1714302624.139014), (23.624, 1714302624.636938), (24.95, 1714302632.278905),
        (8.645, 1714302634.153236), (6.848, 1714302634.654161), (18.894, 1714302638.741316), (16.462, 1714302638.95521), (19.03, 1714302644.136743), (3.62, 1714302644.140954), (24.315, 1714302644.639754), (24.048, 1714302648.769602),
        (23.22, 1714302652.221242), (24.33, 1714302654.29856), (23.492, 1714302654.649078), (20.292, 1714302664.138584), (0.372, 1714302664.14164), (4.579, 1714302664.65042), (0.279, 1714302664.661035), (23.322, 1714302674.144138),
        (3.884, 1714302674.654652), (19.237, 1714302684.308341), (2.738, 1714302684.312243), (23.241, 1714302684.372143), (23.801, 1714302684.635012), (23.496, 1714302694.137047), (23.07, 1714302694.162828), (23.505, 1714302694.635948),
        (1.102, 1714302704.138258), (26.157, 1714302714.257491), (11.874, 1714302714.907589), (8.732, 1714302724.648325), (13.871, 1714302732.643522), (16.581, 1714302734.196834), (23.879, 1714302734.637089), (23.356, 1714302734.665525)
    ]
    key = ('TCP', 'ICMP')
    data = data2
    for rtt, timestamp in data:
        analyesr.add(rtt, timestamp)
    analyesr.__print__()
if __name__ == "__main__":
    analyser_test()


