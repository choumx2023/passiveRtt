import numpy as np
import matplotlib.pyplot as plt
from decimal import Decimal
from mpl_toolkits.mplot3d import Axes3D
import sys
import os
import pickle
import math
import copy
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.RttTable import RTTTable
from src.Monitor import NetworkTrafficMonitor, CompressedIPNode, CompressedIPTrie   
import argparse
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
def read_pickle(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)
class MonitorEval:
    def __init__(self, current_monitor : NetworkTrafficMonitor, new_monitor : NetworkTrafficMonitor):
        self.current_monitor = current_monitor
        self.new_monitor = new_monitor
    def plot_rtt(data : NetworkTrafficMonitor):
        MonitorEval.plot_monitor_hierachy(data.ipv4_trie)
        MonitorEval.plot_monitor_hierachy(data.ipv6_trie)

    
    @staticmethod
    def compare_monitors(monitor1 : NetworkTrafficMonitor, monitor2 : NetworkTrafficMonitor):
        monitor1_ipv4, monitor1_ipv6 = [], []
        monitor2_ipv4, monitor2_ipv6 = [], []
        
            
    def plot_monitor_hierachy(data : CompressedIPTrie):
        pass
        
def main():
    current_monitor = read_pickle('../data/current_monitor.pkl')
    new_tcp_monitor = read_pickle('../data/new_tcptrace_monitor.pkl')
    monitor_eval = MonitorEval(current_monitor, new_tcp_monitor)
    MonitorEval.plot_rtt(current_monitor)
    MonitorEval.plot_rtt(new_tcp_monitor)
if __name__ == '__main__':
    main()