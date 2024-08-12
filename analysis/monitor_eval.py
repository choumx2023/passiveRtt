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
    def __init__(self):
        baseline = read_pickle('baseline.pkl')
        tcptrace = read_pickle('tcptrace.pkl')
        current = read_pickle('current.pkl')
    def plot_rtt(data : NetworkTrafficMonitor):
        MonitorEval.plot_monitor_hierachy(data.ipv4_trie)
        MonitorEval.plot_monitor_hierachy(data.ipv6_trie)
    def plot_monitor_hierachy(data : CompressedIPTrie):
        
        
