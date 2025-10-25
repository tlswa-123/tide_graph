#!/usr/bin/env python3
"""
声音可视化游戏 - 自动演示脚本
用于录制演示视频或展示游戏功能
"""

import pygame
import numpy as np
import math
import time
import os
import sys

# 添加当前目录到path以便导入游戏模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

class GameDemo:
    """游戏演示类 - 自动播放和演示功能"""
    
    def __init__(self):
        self.demo_sounds = [
            # 演示用的模拟声音数据 (volume, frequency, duration)
            (0.02, 120, 0.5),   # 低频海洋方块
            (0.05, 180, 0.8),   # 中频沙漠方块  
            (0.08, 250, 1.0),   # 高频草地方块
            (0.03, 100, 0.6),   # 另一个海洋方块
            (0.06, 220, 0.7),   # 另一个草地方块
            (0.04, 160, 0.9),   # 沙漠方块
            (0.07, 280, 1.2),   # 草地方块
        ]
        self.demo_clicks = [
            # 演示用的点击位置 (screen_x, screen_y, delay)
            (600, 400, 2.0),   # 第一个点击
            (650, 350, 2.5),   # 第二个点击  
            (700, 300, 2.0),   # 第三个点击
            (750, 250, 2.5),   # 第四个点击
            (800, 200, 2.0),   # 第五个点击
            (850, 150, 2.5),   # 最后一个点击到达目标
        ]
        self.current_sound_index = 0
        self.current_click_index = 0
        self.demo_start_time = 0
        self.last_action_time = 0
        self.demo_mode = True
        
    def get_demo_sound(self):
        """获取演示用的模拟声音数据"""
        if self.current_sound_index >= len(self.demo_sounds):
            return None
            
        sound_data = self.demo_sounds[self.current_sound_index]
        self.current_sound_index += 1
        
        return {
            'volume': sound_data[0],
            'frequency': sound_data[1], 
            'duration': sound_data[2]
        }
    
    def get_demo_click(self, current_time):
        """获取演示用的点击位置"""
        if self.current_click_index >= len(self.demo_clicks):
            return None
            
        click_data = self.demo_clicks[self.current_click_index]
        
        # 检查是否到了点击时间
        if current_time - self.last_action_time >= click_data[2]:
            self.current_click_index += 1
            self.last_action_time = current_time
            return (click_data[0], click_data[1])
        
        return None
    
    def print_demo_instructions(self):
        """打印演示说明"""
        print("🎬 声音可视化游戏 - 自动演示模式")
        print("=" * 50)
        print("📝 演示内容：")
        print("   1. 自动生成不同类型的方块（海洋、沙漠、草地）")
        print("   2. 模拟声音触发和鼠标点击")  
        print("   3. 展示小人移动和到达终点")
        print("   4. 显示游戏胜利画面")
        print()
        print("🎥 录屏建议：")
        print("   - 使用 OBS Studio 或系统自带录屏")
        print("   - 录制 30-60 秒完整游戏流程")
        print("   - 确保音频解说清晰")
        print("   - 展示声音控制的实时效果")
        print()
        print("▶️  按回车开始演示...")
        input()

class DemoAudioProcessor:
    """演示用音频处理器 - 使用预设数据"""
    
    def __init__(self, demo):
        self.demo = demo
        self.is_running = False
        self.last_sound_time = 0
        
    def start(self):
        self.is_running = True
        print("🎤 演示音频处理器启动（使用模拟音频数据）")
        return True
        
    def stop(self):
        self.is_running = False
        
    def get_new_sound(self):
        """模拟声音检测 - 返回演示数据"""
        current_time = time.time()
        
        # 每3秒生成一个新声音
        if current_time - self.last_sound_time >= 3.0:
            sound_data = self.demo.get_demo_sound()
            if sound_data:
                self.last_sound_time = current_time
                print(f"🎵 演示声音: {sound_data['frequency']:.0f}Hz, "
                      f"音量: {sound_data['volume']:.3f}, "
                      f"时长: {sound_data['duration']:.1f}s")
                return sound_data
        
        return None
        
    def get_audio_features(self):
        """返回当前音频特征"""
        return {
            'volume': 0.01,
            'frequency': 200.0,
            'duration': 0.5
        }

def patch_game_for_demo():
    """修改游戏以支持演示模式"""
    try:
        # 导入游戏主类
        from single_block_game import SingleBlockVisualizationGame
        
        # 创建演示控制器
        demo = GameDemo()
        demo.print_demo_instructions()
        
        # 创建游戏实例
        game = SingleBlockVisualizationGame()
        
        # 替换音频处理器为演示版本
        game.audio_processor = DemoAudioProcessor(demo)
        
        # 修改鼠标事件处理以支持自动点击
        original_handle_mouse_click = game.handle_mouse_click
        
        def demo_handle_mouse_click(screen_pos):
            return original_handle_mouse_click(screen_pos)
            
        game.handle_mouse_click = demo_handle_mouse_click
        
        # 在游戏循环中添加自动点击
        original_run = game.run
        
        def demo_run():
            if not game.audio_processor.start():
                print("无法启动演示音频处理器!")
                return
                
            print("🎬 演示开始！展示声音可视化游戏的完整流程...")
            demo.demo_start_time = time.time()
            demo.last_action_time = demo.demo_start_time
            
            try:
                while game.running:
                    dt = game.clock.tick(60) / 1000.0
                    game.game_time += dt
                    
                    # 处理事件
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            game.running = False
                        elif event.type == pygame.KEYDOWN:
                            if event.key == pygame.K_ESCAPE:
                                game.running = False
                    
                    # 自动点击演示
                    current_time = time.time()
                    demo_click = demo.get_demo_click(current_time - demo.demo_start_time)
                    if demo_click and game.game_state == "playing":
                        # 模拟声音检测
                        game.recent_sound_detected = True
                        game.last_sound_time = game.game_time
                        # 执行点击
                        game.handle_mouse_click(demo_click)
                        print(f"🖱️  自动点击位置: {demo_click}")
                    
                    # 更新游戏状态
                    game.update_camera(dt)
                    game.update_blocks(dt) 
                    game.update_game_state(dt)
                    
                    # 渲染
                    game.draw_enhanced_background()
                    
                    camera_offset = game.get_camera_offset()
                    
                    # 绘制方块
                    sorted_blocks = sorted(game.blocks, key=lambda b: b.x + b.z)
                    for block in sorted_blocks:
                        if block.height > 0:
                            game.draw_3d_block(game.screen, block, camera_offset)
                    
                    # 绘制游戏元素
                    if game.game_mode == "platformer":
                        game.player.update(dt)
                        game.goal.update(dt) 
                        game.goal.draw(game.screen, game)
                        game.player.draw(game.screen, game)
                    
                    # 绘制UI
                    game.draw_ui(game.screen)
                    
                    # 绘制演示信息
                    demo_text = game.font.render("🎬 DEMO MODE - Auto-playing", True, (255, 255, 0))
                    game.screen.blit(demo_text, (10, 10))
                    
                    # 更新消息
                    game.update_no_sound_message(dt)
                    game.draw_no_sound_message(game.screen)
                    game.update_game_end_message(dt) 
                    game.draw_game_end_message(game.screen)
                    
                    pygame.display.flip()
                    
                    # 演示完成后自动退出
                    if demo.current_click_index >= len(demo.demo_clicks) and game.game_state in ["won", "lost"]:
                        print("🎉 演示完成！可以停止录屏了。")
                        time.sleep(3)  # 显示结果3秒
                        game.running = False
                        
            except KeyboardInterrupt:
                print("\n演示被中断")
            finally:
                game.audio_processor.stop()
                pygame.quit()
                
        # 替换run方法
        game.run = demo_run
        
        return game
        
    except ImportError as e:
        print(f"❌ 无法导入游戏模块: {e}")
        print("请确保在 sound_visualization_game 目录中运行此脚本")
        return None

def main():
    """主函数"""
    print("🎮 声音可视化游戏 - 演示脚本")
    print("用于录制演示视频和展示功能")
    print()
    
    game = patch_game_for_demo()
    if game:
        game.run()
    else:
        print("演示启动失败")

if __name__ == "__main__":
    main()