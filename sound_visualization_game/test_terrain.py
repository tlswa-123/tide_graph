#!/usr/bin/env python3
"""
地形类型测试脚本 - 验证频率阈值调整
"""

def test_terrain_classification():
    """测试地形分类逻辑"""
    
    print("🌍 地形分类测试")
    print("=" * 50)
    
    # 测试不同频率对应的地形类型
    test_frequencies = [100, 120, 139, 140, 150, 180, 199, 200, 220, 250, 300, 350, 400]
    
    for freq in test_frequencies:
        if freq < 140:  # 海洋
            terrain = "🌊 Ocean"
        elif freq < 200:  # 沙漠  
            terrain = "🏜️  Desert"
        else:  # 草地
            terrain = "🌱 Grassland"
        
        print(f"频率: {freq:3d}Hz → {terrain}")
    
    print("\n📊 阈值说明:")
    print("🌊 海洋 (Ocean):   < 140Hz  (低音)")
    print("🏜️  沙漠 (Desert):  140-200Hz (中音)")  
    print("🌱 草地 (Grassland): > 200Hz  (高音)")
    print("\n✅ 草地阈值已从300Hz降低到200Hz，预期会有更多草地方块！")

if __name__ == "__main__":
    test_terrain_classification()