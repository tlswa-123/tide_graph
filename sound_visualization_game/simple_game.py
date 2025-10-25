"""
简化版声音可视化游戏启动器
用于测试基础功能
"""

import pygame
import numpy as np
import math
import time
from typing import List, Tuple

# 如果没有音频设备，使用模拟音频数据
try:
    import pyaudio
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False
    print("PyAudio not available, using simulated audio data")

class SimulatedAudio:
    """模拟音频数据类"""
    
    def __init__(self):
        self.volume = 0.0
        self.dominant_freq = 440.0
        self.time = 0.0
    
    def start(self):
        pass
    
    def stop(self):
        pass
    
    def update(self, dt):
        """更新模拟音频数据"""
        self.time += dt
        # 模拟音量变化（呼吸效果）
        self.volume = (math.sin(self.time * 2) + 1) * 0.3
        # 模拟频率变化
        self.dominant_freq = 440 + math.sin(self.time * 0.5) * 200

class TerrainBlock:
    """地形方块类"""
    
    def __init__(self, x, z, height=0, decoration_type=0):
        self.x = x
        self.z = z
        self.height = height
        self.decoration_type = decoration_type
        self.target_height = height
        self.color = self._get_color()
    
    def _get_color(self):
        """根据装饰类型获取颜色"""
        colors = {
            0: (150, 180, 200),  # 浅蓝灰
            1: (100, 200, 100),  # 绿色（树）
            2: (230, 180, 130),  # 橙色（房子）
            3: (180, 130, 230)   # 紫色（塔）
        }
        return colors.get(self.decoration_type, (200, 200, 200))
    
    def update_from_audio(self, volume, frequency):
        """根据音频更新方块"""
        self.target_height = max(10, volume * 100)
        
        if frequency < 200:
            self.decoration_type = 0
        elif frequency < 500:
            self.decoration_type = 1
        elif frequency < 1000:
            self.decoration_type = 2
        else:
            self.decoration_type = 3
        
        self.color = self._get_color()
    
    def update_height(self, dt):
        """平滑更新高度"""
        if abs(self.height - self.target_height) > 0.5:
            self.height += (self.target_height - self.height) * dt * 5

class Camera:
    """简化的摄像机类"""
    
    def __init__(self):
        self.view_angle = 0  # 0=东南, 1=西南, 2=西北, 3=东北
        self.transition_progress = 1.0
        self.target_angle = 0
    
    def update_view_angle(self, angle):
        if angle != self.view_angle:
            self.target_angle = angle
            self.transition_progress = 0.0
    
    def update(self, dt):
        if self.transition_progress < 1.0:
            self.transition_progress = min(self.transition_progress + dt * 2, 1.0)
            if self.transition_progress >= 1.0:
                self.view_angle = self.target_angle
    
    def get_offset(self):
        """获取视角偏移"""
        angles = [(100, 50), (-100, 50), (-100, -50), (100, -50)]
        if self.transition_progress < 1.0:
            start_offset = angles[self.view_angle]
            end_offset = angles[self.target_angle]
            progress = 1 - (1 - self.transition_progress) ** 3
            return (
                start_offset[0] + (end_offset[0] - start_offset[0]) * progress,
                start_offset[1] + (end_offset[1] - start_offset[1]) * progress
            )
        return angles[self.view_angle]

class Character:
    """小人角色"""
    
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y
        self.target_x = x
        self.target_y = y
        self.speed = 50
    
    def set_target(self, x, y):
        self.target_x = x
        self.target_y = y
    
    def update(self, dt, terrain_height):
        # 移动到目标位置
        dx = self.target_x - self.x
        dy = self.target_y - self.y
        distance = math.sqrt(dx*dx + dy*dy)
        
        if distance > 1:
            move_dist = self.speed * dt
            if move_dist > distance:
                move_dist = distance
            self.x += (dx / distance) * move_dist
            self.y += (dy / distance) * move_dist

class SimpleVisualizationGame:
    """简化版游戏类"""
    
    def __init__(self, width=1200, height=800):
        pygame.init()
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("声音可视化游戏 - 简化版")
        
        self.width = width
        self.height = height
        self.clock = pygame.time.Clock()
        self.running = True
        
        # 初始化组件
        if AUDIO_AVAILABLE:
            from main import AudioProcessor
            self.audio = AudioProcessor()
        else:
            self.audio = SimulatedAudio()
        
        self.camera = Camera()
        self.terrain = self._create_terrain(15, 15)
        self.character = Character(width//2, height//2)
        
        # UI按钮
        self.buttons = self._create_buttons()
        self.font = pygame.font.Font(None, 36)
    
    def _create_terrain(self, width, height):
        """创建地形"""
        terrain = []
        for x in range(width):
            row = []
            for z in range(height):
                block = TerrainBlock(x - width//2, z - height//2)
                row.append(block)
            terrain.append(row)
        return terrain
    
    def _create_buttons(self):
        """创建按钮"""
        button_size = 60
        margin = 20
        return {
            'left': pygame.Rect(self.width - 2*button_size - 2*margin, 
                               self.height - button_size - margin, 
                               button_size, button_size),
            'right': pygame.Rect(self.width - button_size - margin, 
                                self.height - button_size - margin, 
                                button_size, button_size)
        }
    
    def _draw_isometric_block(self, screen, x, y, height, color, camera_offset):
        """绘制等轴测投影的方块"""
        # 等轴测投影转换
        iso_x = (x - y) * 20 + self.width // 2 + camera_offset[0]
        iso_y = (x + y) * 10 + self.height // 2 + camera_offset[1]
        
        if height > 0:
            # 顶面
            top_points = [
                (iso_x, iso_y - height),
                (iso_x + 20, iso_y - 10 - height),
                (iso_x, iso_y - 20 - height),
                (iso_x - 20, iso_y - 10 - height)
            ]
            pygame.draw.polygon(screen, color, top_points)
            
            # 左面
            left_points = [
                (iso_x - 20, iso_y - 10 - height),
                (iso_x, iso_y - 20 - height),
                (iso_x, iso_y - 20),
                (iso_x - 20, iso_y - 10)
            ]
            darker_color = tuple(max(0, c - 30) for c in color)
            pygame.draw.polygon(screen, darker_color, left_points)
            
            # 右面
            right_points = [
                (iso_x, iso_y - 20 - height),
                (iso_x + 20, iso_y - 10 - height),
                (iso_x + 20, iso_y - 10),
                (iso_x, iso_y - 20)
            ]
            darkest_color = tuple(max(0, c - 50) for c in color)
            pygame.draw.polygon(screen, darkest_color, right_points)
    
    def _draw_decoration(self, screen, block, camera_offset):
        """绘制装饰物"""
        if block.decoration_type == 0:
            return
        
        iso_x = (block.x - block.z) * 20 + self.width // 2 + camera_offset[0]
        iso_y = (block.x + block.z) * 10 + self.height // 2 + camera_offset[1]
        
        if block.decoration_type == 1:  # 树
            # 树干
            trunk_color = (101, 67, 33)
            pygame.draw.rect(screen, trunk_color, 
                           (iso_x - 3, iso_y - block.height - 15, 6, 15))
            # 树冠
            crown_color = (34, 139, 34)
            pygame.draw.circle(screen, crown_color, 
                             (int(iso_x), int(iso_y - block.height - 15)), 8)
        
        elif block.decoration_type == 2:  # 房子
            house_color = (139, 69, 19)
            pygame.draw.rect(screen, house_color, 
                           (iso_x - 8, iso_y - block.height - 16, 16, 16))
            # 屋顶
            roof_points = [
                (iso_x - 10, iso_y - block.height - 16),
                (iso_x, iso_y - block.height - 26),
                (iso_x + 10, iso_y - block.height - 16)
            ]
            pygame.draw.polygon(screen, (160, 82, 45), roof_points)
        
        elif block.decoration_type == 3:  # 塔
            tower_color = (128, 0, 128)
            for i in range(3):
                width = 6 - i
                height = 8
                pygame.draw.rect(screen, tower_color, 
                               (iso_x - width, iso_y - block.height - (i+1)*height, 
                                width*2, height))
    
    def _draw_character(self, screen, camera_offset):
        """绘制角色"""
        char_x = (self.character.x/20 - self.character.y/20) * 20 + self.width // 2 + camera_offset[0]
        char_y = (self.character.x/20 + self.character.y/20) * 10 + self.height // 2 + camera_offset[1]
        
        # 身体
        pygame.draw.circle(screen, (255, 220, 177), (int(char_x), int(char_y - 15)), 8)
        # 头部
        pygame.draw.circle(screen, (255, 228, 196), (int(char_x), int(char_y - 30)), 6)
    
    def _draw_ui(self, screen):
        """绘制UI"""
        # 绘制按钮
        pygame.draw.rect(screen, (100, 100, 100), self.buttons['left'])
        pygame.draw.rect(screen, (100, 100, 100), self.buttons['right'])
        
        # 按钮文字
        left_text = self.font.render("←", True, (255, 255, 255))
        right_text = self.font.render("→", True, (255, 255, 255))
        
        screen.blit(left_text, (self.buttons['left'].centerx - 10, 
                               self.buttons['left'].centery - 12))
        screen.blit(right_text, (self.buttons['right'].centerx - 10, 
                                self.buttons['right'].centery - 12))
        
        # 显示音频信息
        info_text = f"音量: {self.audio.volume:.2f} | 频率: {self.audio.dominant_freq:.0f}Hz"
        text_surface = self.font.render(info_text, True, (255, 255, 255))
        screen.blit(text_surface, (10, 10))
        
        # 显示视角信息
        view_names = ["东南", "西南", "西北", "东北"]
        view_text = f"视角: {view_names[self.camera.view_angle]}"
        view_surface = self.font.render(view_text, True, (255, 255, 255))
        screen.blit(view_surface, (10, 50))
    
    def _update_terrain_from_audio(self):
        """根据音频更新地形"""
        volume = self.audio.volume
        frequency = self.audio.dominant_freq
        
        center_x = len(self.terrain) // 2
        center_z = len(self.terrain[0]) // 2
        radius = max(1, int(volume * 3))
        
        for dx in range(-radius, radius + 1):
            for dz in range(-radius, radius + 1):
                x = center_x + dx
                z = center_z + dz
                
                if 0 <= x < len(self.terrain) and 0 <= z < len(self.terrain[0]):
                    distance = math.sqrt(dx*dx + dz*dz)
                    if distance <= radius:
                        influence = max(0, 1 - distance / radius)
                        adjusted_volume = volume * influence
                        self.terrain[x][z].update_from_audio(adjusted_volume, frequency)
    
    def _handle_click(self, pos):
        """处理点击事件"""
        if self.buttons['left'].collidepoint(pos):
            new_angle = (self.camera.view_angle - 1) % 4
            self.camera.update_view_angle(new_angle)
        elif self.buttons['right'].collidepoint(pos):
            new_angle = (self.camera.view_angle + 1) % 4
            self.camera.update_view_angle(new_angle)
        else:
            # 点击地面移动角色
            self.character.set_target(pos[0] - self.width//2, pos[1] - self.height//2)
    
    def run(self):
        """主游戏循环"""
        self.audio.start()
        
        try:
            while self.running:
                dt = self.clock.tick(60) / 1000.0
                
                # 处理事件
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.running = False
                    elif event.type == pygame.MOUSEBUTTONDOWN:
                        if event.button == 1:
                            self._handle_click(event.pos)
                
                # 更新模拟音频（如果使用模拟数据）
                if not AUDIO_AVAILABLE:
                    self.audio.update(dt)
                
                # 更新游戏状态
                self.camera.update(dt)
                self._update_terrain_from_audio()
                
                # 更新地形
                for row in self.terrain:
                    for block in row:
                        block.update_height(dt)
                
                # 更新角色
                self.character.update(dt, 0)
                
                # 渲染
                self.screen.fill((135, 206, 235))  # 天空蓝
                
                camera_offset = self.camera.get_offset()
                
                # 绘制地形（从后往前绘制）
                for z in range(len(self.terrain[0]) - 1, -1, -1):
                    for x in range(len(self.terrain)):
                        block = self.terrain[x][z]
                        if block.height > 0:
                            self._draw_isometric_block(self.screen, block.x, block.z, 
                                                     block.height, block.color, camera_offset)
                            self._draw_decoration(self.screen, block, camera_offset)
                
                # 绘制角色
                self._draw_character(self.screen, camera_offset)
                
                # 绘制UI
                self._draw_ui(self.screen)
                
                pygame.display.flip()
                
        finally:
            self.audio.stop()
            pygame.quit()

if __name__ == "__main__":
    game = SimpleVisualizationGame()
    game.run()