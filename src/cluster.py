import math
import numpy as np
import copy
from collections import deque
import ruptures as rpt
class IntervalMerger:
    def __init__(self, rtt_threshold=5, rtt_split_threshold=20):
        """
        初始化区间合并类
        :param rtt_threshold: 用于判断 rtt 是否接近的阈值
        :param rtt_split_threshold: RTT 差异超过该值时触发拆分逻辑
        """
        self.rtt_threshold = rtt_threshold
        self.rtt_split_threshold = rtt_split_threshold
        self.intervals = []

    def add_interval(self, rtt, min_val, max_val):
        """
        添加一个新的区间，如果满足条件，则合并或拆分
        :param rtt: 当前区间的 rtt 值
        :param min_val: 当前区间的最小值
        :param max_val: 当前区间的最大值
        """
        # 检查是否需要拆分
        for i, interval in enumerate(self.intervals):
            if (
                min_val >= interval["min"]
                and max_val <= interval["max"]
                and abs(interval["rtt"] - rtt) > self.rtt_split_threshold
            ):
                # 拆分区间
                new_intervals = []
                if min_val > interval["min"]:
                    new_intervals.append({
                        "rtt": interval["rtt"],
                        "min": interval["min"],
                        "max": min_val,
                    })
                new_intervals.append({
                    "rtt": rtt,
                    "min": min_val,
                    "max": max_val,
                })
                if max_val < interval["max"]:
                    new_intervals.append({
                        "rtt": interval["rtt"],
                        "min": max_val,
                        "max": interval["max"],
                    })
                # 替换老区间并检查是否重复
                self.intervals.pop(i)
                self.intervals.extend(new_intervals)
                break

        else:
            # 合并区间逻辑
            merged = False
            for interval in self.intervals:
                if (
                    abs(interval["rtt"] - rtt) <= self.rtt_threshold
                    and not (max_val < interval["min"] or min_val > interval["max"])
                ):
                    interval["rtt"] = min(interval["rtt"], rtt)
                    interval["min"] = min(interval["min"], min_val)
                    interval["max"] = max(interval["max"], max_val)
                    merged = True
                    break

            if not merged:
                # 添加新区间
                self.intervals.append({"rtt": rtt, "min": min_val, "max": max_val})

        # 删除重复的区间
        self._remove_duplicates()
        # 自动合并相邻区间
        self._merge_adjacent_intervals()

    def _remove_duplicates(self):
        """
        删除重复的区间，确保没有相同的区间存在
        """
        seen = []
        unique_intervals = []
        for interval in self.intervals:
            if interval not in seen:
                unique_intervals.append(interval)
                seen.append(interval)
        self.intervals = unique_intervals

    def _merge_adjacent_intervals(self):
        """
        自动合并相邻或重叠的区间
        """
        # 排序区间，按照最小值排序
        self.intervals.sort(key=lambda x: x["min"])
        
        merged_intervals = []
        for interval in self.intervals:
            if not merged_intervals:
                merged_intervals.append(interval)
            else:
                last = merged_intervals[-1]
                # 如果当前区间与最后一个区间重叠或接触，则合并
                if (last["rtt"] - self.rtt_threshold <= interval["rtt"] <= last["rtt"] + self.rtt_threshold) and (last["max"] >= interval["min"]):
                    last["min"] = min(last["min"], interval["min"])
                    last["max"] = max(last["max"], interval["max"])
                    last["rtt"] = min(last["rtt"], interval["rtt"])
                else:
                    merged_intervals.append(interval)
        self.intervals = merged_intervals

    def get_intervals(self):
        """
        返回所有合并后的区间
        """
        return self.intervals

    def print(self):
        """
        打印所有区间
        """
        for interval in self.intervals:
            print(f"RTT: {interval['rtt']}, Min: {interval['min']}, Max: {interval['max']}")
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
        return [], [], []
    if len(values) == 1:
        return [values[0], values[0]], [0] , [0]
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
class Cluster:
    def __init__(self):
        self.estimation = []
        self.window = deque(maxlen=100)
        self.window_range = [-1, -1]
        self.anomalies = []
        self.cluster = IntervalMerger()
    def update_window(self, new_value):
        self.window.append(new_value)
        return
    def update_cluster(self, centers, main_cluster, main_range, minor_range, ignore = False):
        if ignore:
            self.cluster.add_interval(centers[main_cluster], main_range[0], main_range[1])
            return
        min_range = -1
        if minor_range == [-1, -1]:
            self.cluster.add_interval(centers[main_cluster], main_range[0], main_range[1])
        elif main_range[0] < minor_range[0] and main_range[1] > minor_range[1]:
            self.cluster.add_interval(centers[main_cluster], main_range[0], minor_range[0])
            self.cluster.add_interval(centers[1 - main_cluster], minor_range[0], minor_range[1])
            self.cluster.add_interval(centers[main_cluster], minor_range[1] , main_range[1])

        elif minor_range[0] < main_range[0] and minor_range[1] > main_range[1]:
            val1, val2, val3 = [], [], []
            for i in self.window:
                if i[1] <= minor_range[0]:
                    val1.append(i[0])
                elif i[1] <= main_range[1]:
                    val2.append(i[0])
                else:
                    val3.append(i[0])
            mean1, mean2, mean3 = sum(val1) / len(val1), sum(val2) / len(val2), sum(val3) / len(val3)
            self.cluster.add_interval(mean1, minor_range[0], main_range[0])
            self.cluster.add_interval(mean2, main_range[1], minor_range[0])
            self.cluster.add_interval(mean3, minor_range[0], minor_range[1])
        elif main_range[0] > minor_range[1] or main_range[1] < minor_range[0]:
            if main_range[0] > minor_range[1]:
                self.cluster.add_interval(centers[1 - main_cluster], minor_range[0], minor_range[1])
                self.cluster.add_interval(centers[main_cluster], main_range[0], main_range[1])

            else:
                self.cluster.add_interval(centers[main_cluster], main_range[0], main_range[1])
                self.cluster.add_interval(centers[1 - main_cluster], minor_range[0], minor_range[1])

        else:
            if main_range[0] < minor_range[0]:
                self.cluster.add_interval(centers[main_cluster], main_range[0], minor_range[0])
                self.cluster.add_interval(centers[1 - main_cluster], main_range[1], minor_range[1])

            else:
                self.cluster.add_interval(centers[main_cluster], minor_range[0], main_range[0])
                self.cluster.add_interval(centers[1 - main_cluster], minor_range[1], main_range[1])
        if len(self.cluster.intervals) > 1:
            second_interval = self.cluster.intervals[-2]
            r = second_interval['min'] - 1e-4
            while self.window and self.window[0][1] <= r:
                self.window.popleft()
                        
        else:
            return
        
    def add(self, rtt, timestamp):
        self.update_window((rtt, timestamp))
        centers, labels, total_dis = kmeans_1d_with_tolerance([i[0] for i in self.window], max_iter=10, tol=1e-5, verbose=False)
        weights = [labels.count(0)/ len(labels), labels.count(1)/ len(labels)] 
        print(labels)
        ignore = False
        if centers[1] - centers[0] <= centers[0] * 0.25 + 3:
            centers[0] = centers[0] * weights[0] + centers[1] * weights[1]
            labels = [0] * len(labels)
            
        if weights[0] >= weights[1]:
            main_cluster = 0
        else:
            main_cluster = 1
        if weights[main_cluster] < 0.2:
            ignore = True
        main_cluster_set = []
        minor_cluster_set = []
        for i in range(len(labels)):
            if labels[i] == main_cluster:
                main_cluster_set.append(self.window[i][1])
            else:
                minor_cluster_set.append(self.window[i][1])
        main_cluster_range = [min(main_cluster_set), max(main_cluster_set)]
        if minor_cluster_set == []:
            minor_cluster_range = [-1, -1]
        else:
            minor_cluster_range = [min(minor_cluster_set), max(minor_cluster_set)]
        self.update_cluster(centers, main_cluster, main_cluster_range, minor_cluster_range, ignore)
        if labels[-1] in [0 , 1]:
            if weights[labels[-1]] < 0.3:
                self.anomalies.append((rtt, timestamp))
        else:
            self.anomalies.append((rtt, timestamp))
    def print(self):
        for i in self.estimation:
            print(i)
        print('Main Cluster:')
        print(self.cluster.print())
        print('Anomalies:')
        print(self.anomalies)
        return


if __name__ == "__main__":
    data = [            
            (11.392, 1718744401.684368), (11.675, 1718744402.756191), (12.35, 1718744409.940866), (11.91, 1718744414.240781), (35.94, 1718744446.554092), (11.089, 1718744446.840014), (10.347, 1718744446.843904), (16.203, 1718744448.410876),
            (16.886, 1718744448.56633), (10.898, 1718744448.698746), (10.473, 1718744448.854102), (10.737, 1718744448.997406), (25.236, 1718744449.16789), (14.175, 1718744449.333419), (22.793, 1718744449.389808), (16.06, 1718744449.389986),
            (25.921, 1718744449.393518), (11.131, 1718744449.554957), (10.366, 1718744449.558224), (27.251, 1718744449.658245), (15.225, 1718744449.658533), (20.075, 1718744449.745386), (20.13, 1718744449.745449), (16.833, 1718744449.74562),
            (17.618, 1718744449.746468), (24.136, 1718744449.765859), (28.39, 1718744449.768849), (28.434, 1718744449.768949), (9.838, 1718744449.776595), (39.903, 1718744449.821927), (23.153, 1718744449.846036), (21.124, 1718744449.846186),
            (21.713, 1718744449.847241), (72.084, 1718744449.848762), (72.27, 1718744449.85026), (73.446, 1718744449.852774), (72.735, 1718744449.853378), (72.977, 1718744449.854946), (75.359, 1718744449.858668), (13.52, 1718744449.893315),
            (9.693, 1718744450.006481), (11.413, 1718744459.608373), (10.997, 1718744507.399179)]
    cluster = Cluster()
    data = [(v[0], v[1] - 1718744400) for v in data]
    for i in data:
        print(i)
        cluster.add(i[0], i[1])
        
    cluster.print()