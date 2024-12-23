import math
import numpy as np
from collections import deque
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
                        if (abs(v - center) > outlier_threshold * avg_distance or abs(v - center) < 1 / outlier_threshold * avg_distance) and abs(v - center) > 10:
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
    total_distance = [0, 0]
    counts = [0, 0]
    for i in values:
        if abs(i - centers[0]) > abs(i - centers[1]):
            total_distance[1] += abs(i - centers[1]) ** 2
            counts[1] += 1
        else:
            total_distance[0] += abs(i - centers[0]) ** 2
            counts[0] += 1
    total_dis =  [math.sqrt(total_distance[l] / (counts[l] if counts[l] != 0 else 1))  for l in range(2)]
    for i, j in enumerate(vs):
        distances = [abs(j - c) for c in centers]
        min_idx = distances.index(min(distances))
        labels[i] = min_idx
        
    return centers, labels, total_dis

import numpy as np
def run_length_analysis(sequence):
    """
    对01序列进行运行长度分析，并给出统计指标。
    
    参数：
        sequence (list of int): 包含0和1的序列，如 [0,1,0,1,1,0]
        
    返回：
        segments (dict): {0: [段长列表], 1: [段长列表]}
        stats (dict): 包含0和1的统计信息，如段数、平均段长、标准差、最大、最小
    """
    if not sequence:
        return {0: [], 1: []}, {}
    
    # 统计每种值连续段的长度
    segments = {0: [], 1: []}
    current_value = sequence[0]
    current_length = 1
    
    for i in range(1, len(sequence)):
        if sequence[i] == current_value:
            current_length += 1
        else:
            # 当前段结束，记录结果
            segments[current_value].append(current_length)
            # 重置新的段
            current_value = sequence[i]
            current_length = 1
    # 别忘了记录最后一段
    segments[current_value].append(current_length)
    
    return segments

def transition_matrix(sequence):
    counts = np.zeros((2, 2))  # 统计转移组合的次数
    n = len(sequence)
    for i in range(n - 1):
        counts[sequence[i], sequence[i + 1]] += 1
    
    

    return counts

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
'''
ICMP  - Response
ICMPv6 - Response
DNS Response
TCP SYN-ACK
TCP Timestamp
TCP <RST>
TCP Normal
TCP PSH
TCP Back-to-back
'''

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
        centers, labels, distance = kmeans_1d_with_tolerance(values_in_window)
        len_values = len(values_in_window)
        transition_result = transition_matrix(labels)
        rank_sum, rank_distance, e  = calculate_rank_sum(centers, labels)
        min1, min2, max1, max2 = rank_distance[2:]
        len1, len2 = [len([l for l in labels if l == i]) for i in range(2)]
        actual = rank_distance[:2]
        len
        def classify_point(actual, min1, max1, min2, max2, len1, len2):
    # 定义分位点
            total = len1 + len2
            interval1 = total / len1
            interval2 = total / len2
            q0 = [min(min1 * 0.8 + max1 * 0.2, min1 * (interval1) ** 2) , min1 * 0.1 + max1 * 0.9]
            q1 = [min(min2 * 0.8 + max2 * 0.2, min2 * (interval2) ** 2 ), min2 * 0.1 + max2 * 0.9]
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
        segments = run_length_analysis(labels)
        # 离心率，不偏，小聚类偏，小聚类和大聚类都偏
        numbers = [len([l for l in labels if l == i]) for i in range(2)]
        weights = [
            len([l for l in labels if l == 0]) / len(labels),
            len([l for l in labels if l == 1]) / len(labels)
        ]
        label = labels[-1]
        len_values = len(labels)
        total_distance = abs(weights[0] * (rank_sum[0] - len(labels)/2 + 0.5) )+ abs(weights[1] * (rank_sum[1] - len(labels)/2 + 0.5) ) / 2
        total_distance *= len_values
        min_weight = min(weights)
        max_weight = max(weights)
        max_num = max(numbers)
        max_distance = abs((len_values - 1) /2 * max_num - (max_num - 1) * max_num/ 2)
        if min(weights) < 0.2 :
            elipse = 'saltation'
        elif total_distance <= max_distance * 0.3 :
            elipse = 'normal'
        elif total_distance <= max_distance * 0.8:
            elipse = 'medium'
        else:
            elipse = 'big'
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
            'elipse': elipse, #zb
            'transition': transition_result,
            'segments': segments,
        }

    def add(self, rtt, timestamp):
        conclusion = self.detect_changes_with_kmeans(rtt, timestamp)
        self.conclusion.append(conclusion)
        e = conclusion['e']
        len_values = len(conclusion['labels'])
        label = conclusion['label']
        centers = conclusion['centers']
        rank_sum = conclusion['rank_sum']
        weights = conclusion['weights']
        distance = conclusion['distance']
        # 缺少数据
        if conclusion['status'] == 'insufficient_data':
            if weights[label] > 0.66:
                self.estimation.append({'con': 'normal', 'rtt': rtt, 'timestamp': timestamp, 'rtt_estimation': centers[label], 'label': label, 'centers': centers, 'weights':weights})
            else:
                self.estimation.append({'con': 'insufficient_data', 'rtt': rtt, 'timestamp': timestamp, 'rtt_estimation': centers[0] * weights[0] + centers[1] * weights[1], 'label': label, 'centers': centers, 'weights':weights})
            return
        elif conclusion['elipse'] == 'saltation':
            if weights[label] > 1 -  0.2 :
                self.estimation.append({'con': 'normal', 'rtt': rtt, 'timestamp': timestamp, 'rtt_estimation': centers[label], 'label': label, 'centers': centers, 'weights':weights})
            else:
                self.estimation.append({'con': 'saltation', 'rtt': rtt, 'timestamp': timestamp, 'rtt_estimation': centers[1 - label], 'label': label, 'centers': centers, 'weights':weights})
            return
        
        # 无偏心
        elif conclusion['elipse'] == 'normal':
            # 如果都是mid，那么就是normal
            if conclusion['result_distance'] == ('mid', 'mid'):
                self.estimation.append({'con': 'normal', 'rtt': rtt, 'timestamp': timestamp, 'rtt_estimation': centers[label], 'label': label, 'centers': centers, 'weights':weights})
            elif ((conclusion['result_distance'] == ('small', 'mid') and weights[0] >= weights[1]) or (conclusion['result_distance'] == ('mid', 'small') and weights[1] >= weights[0]) or conclusion['result_distance'] == ('small', 'small')) and conclusion['segments'][label][-1] > 3:
                self.estimation.append({'con': 'step_change', 'rtt': rtt, 'timestamp': timestamp, 'rtt_estimation': centers[label], 'label': label, 'centers': centers, 'weights':weights})
            else:
                if (conclusion['segments'][label][-1] > conclusion['segments'][1 - label][-1] and conclusion['segments'][label][-1] > 3) or (conclusion['segments'][label][-1] > 0.3 *conclusion['segments'][1 - label][-1] and conclusion['segments'][label][-1] > 6):
                    self.estimation.append({'con': 'turbulent_normal', 'rtt': rtt, 'timestamp': timestamp, 'rtt_estimation': centers[label], 'label': label, 'centers': centers, 'weights':weights})
                elif conclusion['segments'][label][-1] > 0.2 * conclusion['segments'][1 - label][-1]:
                    self.estimation.append({'con': 'turbulent_normal', 'rtt': rtt, 'timestamp': timestamp, 'rtt_estimation': centers[label] * weights[label] + centers[label - 1] * weights[1 - label], 'label': label, 'centers': centers, 'weights':weights})
                else:
                    self.estimation.append({'con': 'turbulent_normal', 'rtt': rtt, 'timestamp': timestamp, 'rtt_estimation': centers[1 - label], 'label': label, 'centers': centers, 'weights':weights})
        # 有小偏心
        elif conclusion['elipse'] == 'medium':
            
            if conclusion['result_distance'] == ('small', 'small'):
                self.estimation.append({'con': 'normal', 'rtt': rtt, 'timestamp': timestamp, 'rtt_estimation': centers[label], 'label': label, 'centers': centers, 'weights':weights})
            elif ((conclusion['result_distance'] == ('small', 'mid') and weights[0] >= weights[1]) or (conclusion['result_distance'] == ('mid', 'small')) and weights[1] >= weights[0]) and conclusion['segments'][label][-1] > 3 :
                self.estimation.append({'con': 'step_change', 'rtt': rtt, 'timestamp': timestamp, 'rtt_estimation': centers[label], 'label': label, 'centers': centers, 'weights':weights})
                
            else:
                if conclusion['segments'][label][-1] > conclusion['segments'][1 - label][-1] and conclusion['segments'][label][-1] > 3:
                    self.estimation.append({'con': 'turbulent_medium', 'rtt': rtt, 'timestamp': timestamp, 'rtt_estimation': centers[label], 'label': label, 'centers': centers, 'weights':weights})
                elif conclusion['segments'][label][-1] > 0.2 * conclusion['segments'][1 - label][-1]:
                    self.estimation.append({'con': 'turbulent_medium', 'rtt': rtt, 'timestamp': timestamp, 'rtt_estimation': centers[label] * weights[label] + centers[label - 1] * weights[1 - label], 'label': label, 'centers': centers, 'weights':weights})
                else:
                    self.estimation.append({'con': 'turbulent_medium', 'rtt': rtt, 'timestamp': timestamp, 'rtt_estimation': centers[1 - label], 'label': label, 'centers': centers, 'weights':weights})
                
        # 有大偏心
        elif conclusion['elipse'] == 'big':
            
            if rtt > centers[1 - label] + conclusion['distance'][1 - label] and abs(centers[1 - label] - centers[label]) < 20 and rank_sum[label] > rank_sum[1 - label] and conclusion['segments'][label][-1] > 3:
                self.estimation.append({'con': 'gradual_increase', 'rtt': rtt, 'timestamp': timestamp, 'rtt_estimation': centers[label], 'label': label, 'centers': centers, 'weights':weights})
                
            elif rtt < centers[1 - label] - conclusion['distance'][1 - label] and abs(centers[label] - centers[1 - label]) < 20 and rank_sum[label] > rank_sum[1 - label] and conclusion['segments'][label][-1] > 3:
                self.estimation.append({'con': 'gradual_decrease', 'rtt': rtt, 'timestamp': timestamp, 'rtt_estimation': centers[label], 'label': label, 'centers': centers, 'weights':weights})
            
            
            elif rtt < centers[label] + conclusion['distance'][label] and rtt > centers[label] - conclusion['distance'][label] and centers[label] - centers[1 -label] > 15 and conclusion['segments'][label][-1] > 3:
                self.estimation.append({'con': 'sudden_increase', 'rtt': rtt, 'timestamp': timestamp, 'rtt_estimation': centers[label], 'label': label, 'centers': centers, 'weights':weights})
            elif rtt < centers[label] + conclusion['distance'][label] and rtt > centers[label] - conclusion['distance'][label] and centers[1 - label] - centers[label] > 15 and conclusion['segments'][label][-1] > 3:
                self.estimation.append({'con': 'sudden_decrease', 'rtt': rtt, 'timestamp': timestamp, 'rtt_estimation': centers[label], 'label': label, 'centers': centers, 'weights':weights})
            
            else:
                
                if (conclusion['segments'][label][-1] > conclusion['segments'][1 - label][-1] and conclusion['segments'][label][-1] > 3) or (conclusion['segments'][label][-1] > 0.3 *conclusion['segments'][1 - label][-1] and conclusion['segments'][label][-1] > 6):
                    self.estimation.append({'con': 'turbulent_big', 'rtt': rtt, 'timestamp': timestamp, 'rtt_estimation': centers[label], 'label': label, 'centers': centers, 'weights':weights})
                elif conclusion['segments'][label][-1] > 0.2 * conclusion['segments'][1 - label][-1]:
                    self.estimation.append({'con': 'turbulent_big', 'rtt': rtt, 'timestamp': timestamp, 'rtt_estimation': centers[label] * weights[label] + centers[1 - label] * weights[1 - label], 'label': label, 'centers': centers, 'weights':weights})
                else:
                    self.estimation.append({'con': 'turbulent_big', 'rtt': rtt, 'timestamp': timestamp, 'rtt_estimation': centers[1 - label], 'label': label, 'centers': centers, 'weights':weights})
        return 
    # 
    '''
    先进行add
    然后进行get_current_estimation
    '''
    def get_current_estimation(self):
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
        current = analyesr.get_current_estimation()
        print(current['rtt'], current['rtt_estimation'], current['con'],current['centers']) 
if __name__ == "__main__":
    analyser_test()
