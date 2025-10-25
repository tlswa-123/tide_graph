#!/usr/bin/env python3
"""测试运行脚本"""

import sys
print(f"Python version: {sys.version}")

try:
    import pygame
    print("pygame imported successfully")
except ImportError as e:
    print(f"pygame import failed: {e}")

try:
    import pyaudio
    print("pyaudio imported successfully")
except ImportError as e:
    print(f"pyaudio import failed: {e}")

try:
    import numpy
    print("numpy imported successfully")
except ImportError as e:
    print(f"numpy import failed: {e}")

try:
    import scipy
    print("scipy imported successfully")
except ImportError as e:
    print(f"scipy import failed: {e}")

print("尝试导入游戏...")
try:
    from single_block_game import SingleBlockVisualizationGame
    print("游戏类导入成功")
    
    print("尝试创建游戏实例...")
    game = SingleBlockVisualizationGame()
    print("游戏创建成功！")
    
except Exception as e:
    print(f"游戏导入/创建失败: {e}")
    import traceback
    traceback.print_exc()