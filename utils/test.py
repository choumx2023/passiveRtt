import sys
import pkg_resources

print("Python 解释器路径:", sys.executable)
try:
    dist = pkg_resources.get_distribution("scapy")
    print("Scapy 安装路径:", dist.location)
except pkg_resources.DistributionNotFound:
    print("Scapy 未安装在当前环境中。")