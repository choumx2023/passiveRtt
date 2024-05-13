from ..src.RttTable import RTTTable
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
class RttAnalyse():
    def __init__(self, RttTable: RTTTable):
        self.RttTable = RttTable.rtt_samples
        stats = {}
        for key, samples in self.RttTable.items():
            if samples:
                rtts = [sample['rtt'] for sample in samples]
                avg_rtt = sum(rtts) / len(rtts)
                min_rtt = min(rtts)
                max_rtt = max(rtts)
                rtts_sorted = sorted(rtts)
                median_rtt = rtts_sorted[len(rtts_sorted) // 2]
                stats[key] = {
                    'average': avg_rtt,
                    'median': median_rtt,
                    'min': min_rtt,
                    'max': max_rtt
                }
        return stats
    def get_all_samples(self):
        """收集所有IP对的RTT样本"""
        all_samples = []
        for samples in self.rtt_samples.values():
            all_samples.extend([sample['rtt'] for sample in samples])
        return all_samples

    def plot_cdf(self):
        """生成并显示所有RTT样本的CDF图"""
        data = self.get_all_samples()
        data_sorted = np.sort(data)
        p = 1. * np.arange(len(data)) / (len(data) - 1)  # 计算CDF值
        plt.figure(figsize=(8, 4))
        plt.step(data_sorted, p, where='post')
        plt.title('CDF of RTT Samples')
        plt.xlabel('RTT (ms)')
        plt.ylabel('CDF')
        plt.grid(True)
        plt.show()
    def plot_time_series(self):
        """生成并显示RTT样本随时间的变化图"""
        timestamps = []
        rtts = []
        for samples in self.rtt_samples.values():
            for sample in samples:
                # 假设时间戳已经是UNIX时间戳格式
                timestamps.append(datetime.fromtimestamp(sample['timestamp']))
                rtts.append(sample['rtt'])
        
        # 按时间戳排序数据
        timestamps, rtts = zip(*sorted(zip(timestamps, rtts)))

        plt.figure(figsize=(10, 5))
        plt.plot(timestamps, rtts, marker='o', linestyle='-')
        plt.title('RTT Time Series')
        plt.xlabel('Time')
        plt.ylabel('RTT (ms)')
        plt.grid(True)
        
        # 格式化日期显示
        plt.gca().xaxis.set_major_locator(mdates.MinuteLocator())
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        plt.gcf().autofmt_xdate()  # 自动调整日期显示的角度

        plt.show()
    
