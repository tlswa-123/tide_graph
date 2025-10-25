"""
增强版声音可视化游戏
添加了更多纪念碑谷风格的美化效果
"""

import pygame
import numpy as np
import math
import time
from typing import List, Tuple
import colorsys

# 尝试导入音频库
try:
    import pyaudio
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False
    print("PyAudio not available, using simulated audio data")


class SimulatedAudio:
    """增强的模拟音频数据类"""
    
    def __init__(self):
        self.volume = 0.0
        self.dominant_freq = 440.0
        self.time = 0.0
        self.noise_time = 0.0
        
    def start(self):
        pass
    
    def stop(self):
        pass
    
    def update(self, dt):
        """更新模拟音频数据，增加更多变化"""
        self.time += dt
        self.noise_time += dt * 3.7  # 不同的频率创造更复杂的变化
        
        # 主要音量波形（呼吸效果）
        main_wave = (math.sin(self.time * 1.5) + 1) * 0.5
        # 添加高频噪声
        noise = math.sin(self.noise_time) * 0.1
        # 添加低频调制
        modulation = math.sin(self.time * 0.3) * 0.2
        
        self.volume = max(0, min(1, main_wave + noise + modulation)) * 0.6
        
        # 频率变化更复杂
        base_freq = 440 + math.sin(self.time * 0.7) * 300
        freq_noise = math.sin(self.time * 2.3) * 100
        self.dominant_freq = max(50, base_freq + freq_noise)


class ParticleSystem:
    """粒子系统，用于添加动态效果"""
    
    def __init__(self):
        self.particles = []
    
    def add_particle(self, x, y, color, lifetime=2.0):
        """添加粒子"""
        particle = {
            'x': x,
            'y': y,
            'vx': (np.random.random() - 0.5) * 20,
            'vy': -np.random.random() * 30 - 10,
            'color': color,
            'lifetime': lifetime,
            'max_lifetime': lifetime,
            'size': np.random.randint(2, 6)
        }
        self.particles.append(particle)
    
    def update(self, dt):
        """更新粒子"""
        for particle in self.particles[:]:
            particle['x'] += particle['vx'] * dt
            particle['y'] += particle['vy'] * dt
            particle['vy'] += 50 * dt  # 重力
            particle['lifetime'] -= dt
            
            if particle['lifetime'] <= 0:
                self.particles.remove(particle)
    
    def draw(self, screen):
        """绘制粒子"""
        for particle in self.particles:
            alpha = particle['lifetime'] / particle['max_lifetime']
            color = [int(c * alpha) for c in particle['color']]
            pygame.draw.circle(screen, color, 
                             (int(particle['x']), int(particle['y'])), 
                             int(particle['size'] * alpha))


class TerrainBlock:
    """增强的地形方块类"""
    
    def __init__(self, x, z, height=0, decoration_type=0):
        self.x = x
        self.z = z
        self.height = height
        self.decoration_type = decoration_type
        self.target_height = height
        self.color = self._get_color()
        self.glow_intensity = 0.0
        self.animation_offset = np.random.random() * math.pi * 2
        
    def _get_color(self):
        """根据装饰类型获取纪念碑谷风格的颜色"""
        colors = {
            0: (240, 248, 255),  # 爱丽丝蓝
            1: (144, 238, 144),  # 浅绿色（树）
            2: (255, 218, 185),  # 桃色（房子）
            3: (221, 160, 221)   # 紫罗兰（塔）
        }
        base_color = colors.get(self.decoration_type, (220, 220, 220))
        
        # 添加微妙的色调变化
        hue_shift = (hash(f"{self.x}{self.z}") % 100) / 1000.0
        r, g, b = base_color
        h, s, v = colorsys.rgb_to_hsv(r/255, g/255, b/255)
        h = (h + hue_shift) % 1.0
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        
        return (int(r * 255), int(g * 255), int(b * 255))
    
    def update_from_audio(self, volume, frequency):
        """根据音频更新方块"""
        self.target_height = max(5, volume * 120)
        self.glow_intensity = volume
        
        # 频率映射到装饰类型
        if frequency < 150:
            self.decoration_type = 0
        elif frequency < 400:
            self.decoration_type = 1
        elif frequency < 800:
            self.decoration_type = 2
        else:
            self.decoration_type = 3
        
        self.color = self._get_color()
    
    def update_height(self, dt):
        """平滑更新高度，添加弹性效果"""
        if abs(self.height - self.target_height) > 0.5:
            diff = self.target_height - self.height
            # 弹性动画
            self.height += diff * dt * 8
    
    def get_glow_color(self, time):
        """获取发光颜色"""
        if self.glow_intensity > 0.1:
            # 基于时间的发光动画
            glow_factor = (math.sin(time * 4 + self.animation_offset) + 1) * 0.5
            intensity = self.glow_intensity * glow_factor * 0.3
            
            glow_colors = {
                0: (255, 255, 255),
                1: (50, 255, 50),
                2: (255, 200, 100),
                3: (200, 100, 255)
            }
            
            base_glow = glow_colors.get(self.decoration_type, (255, 255, 255))
            return [int(c * intensity) for c in base_glow]
        return [0, 0, 0]


class Camera:
    """增强的摄像机类，支持更平滑的过渡"""
    
    def __init__(self):
        self.view_angle = 0
        self.transition_progress = 1.0
        self.target_angle = 0
        self.shake_intensity = 0.0
        self.shake_time = 0.0
        
    def update_view_angle(self, angle):
        if angle != self.view_angle:
            self.target_angle = angle
            self.transition_progress = 0.0
    
    def add_shake(self, intensity):
        """添加屏幕震动效果"""
        self.shake_intensity = max(self.shake_intensity, intensity)
    
    def update(self, dt):
        if self.transition_progress < 1.0:
            # 使用缓动函数实现平滑过渡
            self.transition_progress = min(self.transition_progress + dt * 2.5, 1.0)
            if self.transition_progress >= 1.0:
                self.view_angle = self.target_angle
        
        # 更新震动效果
        if self.shake_intensity > 0:
            self.shake_intensity -= dt * 2
            self.shake_time += dt * 30
    
    def get_offset(self):
        """获取视角偏移，包含震动效果"""
        # 基础视角偏移
        angles = [(120, 60), (-120, 60), (-120, -60), (120, -60)]
        
        if self.transition_progress < 1.0:
            start_offset = angles[self.view_angle]
            end_offset = angles[self.target_angle]
            # 使用三次贝塞尔曲线实现平滑过渡
            t = self.transition_progress
            smooth_t = t * t * (3 - 2 * t)
            offset_x = start_offset[0] + (end_offset[0] - start_offset[0]) * smooth_t
            offset_y = start_offset[1] + (end_offset[1] - start_offset[1]) * smooth_t
        else:
            offset_x, offset_y = angles[self.view_angle]
        
        # 添加震动偏移
        if self.shake_intensity > 0:
            shake_x = math.sin(self.shake_time) * self.shake_intensity * 5
            shake_y = math.cos(self.shake_time * 1.3) * self.shake_intensity * 3
            offset_x += shake_x
            offset_y += shake_y
        
        return (offset_x, offset_y)


class Character:
    """增强的角色类，添加动画和特效"""
    
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y
        self.target_x = x
        self.target_y = y
        self.speed = 60
        self.walk_animation = 0.0
        self.is_moving = False
        self.trail_particles = []
        
    def set_target(self, x, y):
        self.target_x = x
        self.target_y = y
        self.is_moving = True
    
    def update(self, dt, particle_system):
        # 移动逻辑
        dx = self.target_x - self.x
        dy = self.target_y - self.y
        distance = math.sqrt(dx*dx + dy*dy)
        
        if distance > 2:
            move_dist = self.speed * dt
            if move_dist > distance:
                move_dist = distance
                self.is_moving = False
            else:
                self.is_moving = True
                
            self.x += (dx / distance) * move_dist
            self.y += (dy / distance) * move_dist
            
            # 行走动画
            self.walk_animation += dt * 10
            
            # 添加足迹粒子
            if int(self.walk_animation) % 20 == 0:
                particle_system.add_particle(
                    self.x, self.y + 10, 
                    (200, 200, 200), 
                    lifetime=1.0
                )
        else:
            self.is_moving = False
    
    def get_animation_offset(self):
        """获取行走动画偏移"""
        if self.is_moving:
            return math.sin(self.walk_animation) * 2
        return 0


class EnhancedVisualizationGame:
    """增强版声音可视化游戏"""
    
    def __init__(self, width=1400, height=900):
        pygame.init()
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("声音可视化游戏 - 纪念碑谷风格 (增强版)")
        
        self.width = width
        self.height = height
        self.clock = pygame.time.Clock()
        self.running = True
        self.game_time = 0.0
        
        # 初始化组件
        if AUDIO_AVAILABLE:
            # 这里可以添加真实的音频处理
            pass
        
        self.audio = SimulatedAudio()
        self.camera = Camera()
        self.terrain = self._create_terrain(18, 18)
        self.character = Character(width//2, height//2)
        self.particle_system = ParticleSystem()
        
        # UI和视觉效果
        self.buttons = self._create_buttons()
        self.font = pygame.font.Font(None, 32)
        self.title_font = pygame.font.Font(None, 48)
        
        # 背景渐变
        self.background_colors = [
            (135, 206, 235),  # 天空蓝
            (255, 182, 193),  # 浅粉色
            (221, 160, 221),  # 紫罗兰
            (173, 216, 230)   # 浅蓝色
        ]
        
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
        """创建美化的按钮"""
        button_size = 70
        margin = 25
        return {
            'left': pygame.Rect(self.width - 2*button_size - 2*margin, 
                               self.height - button_size - margin, 
                               button_size, button_size),
            'right': pygame.Rect(self.width - button_size - margin, 
                                self.height - button_size - margin, 
                                button_size, button_size)
        }
    
    def _draw_gradient_background(self):
        """绘制渐变背景"""
        # 根据音频强度选择背景色
        volume_intensity = self.audio.volume
        color_index = int(volume_intensity * len(self.background_colors))
        color_index = min(color_index, len(self.background_colors) - 1)
        
        current_color = self.background_colors[color_index]
        next_color = self.background_colors[(color_index + 1) % len(self.background_colors)]
        
        # 创建垂直渐变
        for y in range(self.height):
            ratio = y / self.height
            r = int(current_color[0] * (1 - ratio) + next_color[0] * ratio)
            g = int(current_color[1] * (1 - ratio) + next_color[1] * ratio)
            b = int(current_color[2] * (1 - ratio) + next_color[2] * ratio)
            
            pygame.draw.line(self.screen, (r, g, b), (0, y), (self.width, y))
    
    def _draw_isometric_block_enhanced(self, screen, x, y, height, color, camera_offset, glow_color=None):
        """绘制增强的等轴测投影方块"""
        iso_x = (x - y) * 25 + self.width // 2 + camera_offset[0]
        iso_y = (x + y) * 12 + self.height // 2 + camera_offset[1]
        
        if height > 0:
            # 添加发光效果
            if glow_color and any(c > 0 for c in glow_color):
                glow_surface = pygame.Surface((50, int(height + 40)), pygame.SRCALPHA)
                pygame.draw.ellipse(glow_surface, (*glow_color, 30), 
                                  (0, 0, 50, int(height + 40)))
                screen.blit(glow_surface, (iso_x - 25, iso_y - height - 20))
            
            # 顶面 - 添加高光效果
            top_points = [
                (iso_x, iso_y - height),
                (iso_x + 25, iso_y - 12 - height),
                (iso_x, iso_y - 24 - height),
                (iso_x - 25, iso_y - 12 - height)
            ]
            
            # 主颜色
            pygame.draw.polygon(screen, color, top_points)
            
            # 高光
            highlight_color = tuple(min(255, c + 30) for c in color)
            highlight_points = [
                (iso_x - 5, iso_y - height - 5),
                (iso_x + 15, iso_y - 12 - height - 3),
                (iso_x, iso_y - 19 - height - 2),
                (iso_x - 15, iso_y - 9 - height - 4)
            ]
            pygame.draw.polygon(screen, highlight_color, highlight_points)
            
            # 左面
            left_points = [
                (iso_x - 25, iso_y - 12 - height),
                (iso_x, iso_y - 24 - height),
                (iso_x, iso_y - 24),
                (iso_x - 25, iso_y - 12)
            ]
            darker_color = tuple(max(0, c - 40) for c in color)
            pygame.draw.polygon(screen, darker_color, left_points)
            
            # 右面
            right_points = [
                (iso_x, iso_y - 24 - height),
                (iso_x + 25, iso_y - 12 - height),
                (iso_x + 25, iso_y - 12),
                (iso_x, iso_y - 24)
            ]
            darkest_color = tuple(max(0, c - 60) for c in color)
            pygame.draw.polygon(screen, darkest_color, right_points)
            
            # 添加边框
            pygame.draw.polygon(screen, (50, 50, 50), top_points, 1)
            pygame.draw.polygon(screen, (50, 50, 50), left_points, 1)
            pygame.draw.polygon(screen, (50, 50, 50), right_points, 1)
    
    def _draw_decoration_enhanced(self, screen, block, camera_offset):
        """绘制增强的装饰物"""
        if block.decoration_type == 0:
            return
        
        iso_x = (block.x - block.z) * 25 + self.width // 2 + camera_offset[0]
        iso_y = (block.x + block.z) * 12 + self.height // 2 + camera_offset[1]
        
        # 添加动画偏移
        animation_offset = math.sin(self.game_time * 2 + block.animation_offset) * 2
        
        if block.decoration_type == 1:  # 增强的树
            # 树干
            trunk_color = (101, 67, 33)
            trunk_rect = pygame.Rect(iso_x - 4, iso_y - block.height - 20, 8, 20)
            pygame.draw.rect(screen, trunk_color, trunk_rect)
            pygame.draw.rect(screen, (80, 50, 20), trunk_rect, 1)
            
            # 多层树冠
            crown_colors = [(34, 139, 34), (50, 160, 50), (70, 180, 70)]
            for i, crown_color in enumerate(crown_colors):
                radius = 12 - i * 2
                crown_y = iso_y - block.height - 18 + animation_offset - i * 2
                pygame.draw.circle(screen, crown_color, 
                                 (int(iso_x), int(crown_y)), radius)
        
        elif block.decoration_type == 2:  # 增强的房子
            # 房子主体
            house_color = (139, 69, 19)
            house_rect = pygame.Rect(iso_x - 10, iso_y - block.height - 20, 20, 20)
            pygame.draw.rect(screen, house_color, house_rect)
            pygame.draw.rect(screen, (100, 50, 10), house_rect, 1)
            
            # 窗户
            window_color = (255, 255, 100)
            pygame.draw.rect(screen, window_color, 
                           (iso_x - 6, iso_y - block.height - 15, 4, 4))
            pygame.draw.rect(screen, window_color, 
                           (iso_x + 2, iso_y - block.height - 15, 4, 4))
            
            # 屋顶
            roof_points = [
                (iso_x - 12, iso_y - block.height - 20),
                (iso_x, iso_y - block.height - 32 + animation_offset),
                (iso_x + 12, iso_y - block.height - 20)
            ]
            pygame.draw.polygon(screen, (160, 82, 45), roof_points)
            pygame.draw.polygon(screen, (120, 60, 30), roof_points, 1)
        
        elif block.decoration_type == 3:  # 增强的塔
            tower_colors = [(128, 0, 128), (148, 20, 148), (168, 40, 168)]
            for i in range(3):
                width = 8 - i * 2
                height = 10
                tower_y = iso_y - block.height - (i+1)*height + animation_offset * (i + 1) * 0.3
                tower_rect = pygame.Rect(iso_x - width, tower_y, width*2, height)
                pygame.draw.rect(screen, tower_colors[i], tower_rect)
                pygame.draw.rect(screen, (80, 0, 80), tower_rect, 1)
    
    def _draw_character_enhanced(self, screen, camera_offset):
        """绘制增强的角色"""
        char_x = (self.character.x/25 - self.character.y/25) * 25 + self.width // 2 + camera_offset[0]
        char_y = (self.character.x/25 + self.character.y/25) * 12 + self.height // 2 + camera_offset[1]
        
        # 添加行走动画偏移
        walk_offset = self.character.get_animation_offset()
        
        # 阴影
        shadow_color = (0, 0, 0, 50)
        shadow_surface = pygame.Surface((20, 8), pygame.SRCALPHA)
        pygame.draw.ellipse(shadow_surface, shadow_color, (0, 0, 20, 8))
        screen.blit(shadow_surface, (char_x - 10, char_y - 5))
        
        # 身体
        body_color = (255, 220, 177)
        pygame.draw.circle(screen, body_color, 
                          (int(char_x), int(char_y - 18 + walk_offset)), 10)
        pygame.draw.circle(screen, (200, 180, 140), 
                          (int(char_x), int(char_y - 18 + walk_offset)), 10, 2)
        
        # 头部
        head_color = (255, 228, 196)
        pygame.draw.circle(screen, head_color, 
                          (int(char_x), int(char_y - 35 + walk_offset)), 8)
        pygame.draw.circle(screen, (200, 190, 160), 
                          (int(char_x), int(char_y - 35 + walk_offset)), 8, 1)
        
        # 眼睛
        pygame.draw.circle(screen, (0, 0, 0), 
                          (int(char_x - 3), int(char_y - 37 + walk_offset)), 1)
        pygame.draw.circle(screen, (0, 0, 0), 
                          (int(char_x + 3), int(char_y - 37 + walk_offset)), 1)
    
    def _draw_enhanced_ui(self, screen):
        """绘制增强的UI"""
        # 绘制半透明背景
        ui_surface = pygame.Surface((self.width, 100), pygame.SRCALPHA)
        ui_surface.fill((0, 0, 0, 100))
        screen.blit(ui_surface, (0, 0))
        
        # 标题
        title_text = self.title_font.render("纪念碑谷风格声音可视化", True, (255, 255, 255))
        screen.blit(title_text, (20, 10))
        
        # 音频信息，带可视化条
        volume_text = f"音量: {self.audio.volume:.2f}"
        freq_text = f"频率: {self.audio.dominant_freq:.0f}Hz"
        
        text_y = 55
        volume_surface = self.font.render(volume_text, True, (255, 255, 255))
        screen.blit(volume_surface, (20, text_y))
        
        # 音量可视化条
        bar_x = 150
        bar_width = 200
        bar_height = 20
        pygame.draw.rect(screen, (50, 50, 50), (bar_x, text_y, bar_width, bar_height))
        pygame.draw.rect(screen, (100, 255, 100), 
                        (bar_x, text_y, int(bar_width * self.audio.volume), bar_height))
        
        freq_surface = self.font.render(freq_text, True, (255, 255, 255))
        screen.blit(freq_surface, (380, text_y))
        
        # 视角信息
        view_names = ["东南视角", "西南视角", "西北视角", "东北视角"]
        view_text = view_names[self.camera.view_angle]
        view_surface = self.font.render(view_text, True, (255, 255, 255))
        screen.blit(view_surface, (self.width - 200, 20))
        
        # 增强的按钮
        for button_name, button_rect in self.buttons.items():
            # 按钮背景
            button_color = (70, 70, 70, 200)
            button_surface = pygame.Surface((button_rect.width, button_rect.height), pygame.SRCALPHA)
            pygame.draw.rect(button_surface, button_color, (0, 0, button_rect.width, button_rect.height))
            pygame.draw.rect(button_surface, (150, 150, 150), (0, 0, button_rect.width, button_rect.height), 2)
            screen.blit(button_surface, button_rect)
            
            # 按钮图标
            icon = "◀" if button_name == 'left' else "▶"
            icon_surface = self.title_font.render(icon, True, (255, 255, 255))
            icon_rect = icon_surface.get_rect(center=button_rect.center)
            screen.blit(icon_surface, icon_rect)
        
        # 说明文字
        help_text = "点击地面移动小人 | 点击箭头切换视角 | 发出声音看效果"
        help_surface = self.font.render(help_text, True, (255, 255, 255))
        help_rect = help_surface.get_rect()
        help_rect.centerx = self.width // 2
        help_rect.bottom = self.height - 10
        screen.blit(help_surface, help_rect)
    
    def _update_terrain_from_audio(self):
        """根据音频更新地形"""
        volume = self.audio.volume
        frequency = self.audio.dominant_freq
        
        center_x = len(self.terrain) // 2
        center_z = len(self.terrain[0]) // 2
        radius = max(2, int(volume * 6))
        
        # 高音量时添加屏幕震动
        if volume > 0.7:
            self.camera.add_shake((volume - 0.7) * 2)
        
        for dx in range(-radius, radius + 1):
            for dz in range(-radius, radius + 1):
                x = center_x + dx
                z = center_z + dz
                
                if 0 <= x < len(self.terrain) and 0 <= z < len(self.terrain[0]):
                    distance = math.sqrt(dx*dx + dz*dz)
                    if distance <= radius:
                        # 更复杂的影响计算
                        influence = max(0, 1 - (distance / radius) ** 1.5)
                        adjusted_volume = volume * influence
                        
                        # 添加时间偏移创造波浪效果
                        time_offset = math.sin(self.game_time * 3 + distance * 0.5) * 0.1
                        adjusted_volume = max(0, adjusted_volume + time_offset)
                        
                        self.terrain[x][z].update_from_audio(adjusted_volume, frequency)
                        
                        # 在高能量区域生成粒子
                        if adjusted_volume > 0.5 and np.random.random() < 0.1:
                            world_x = (dx - dz) * 25 + self.width // 2
                            world_y = (dx + dz) * 12 + self.height // 2
                            particle_color = self.terrain[x][z].color
                            self.particle_system.add_particle(world_x, world_y, particle_color)
    
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
                self.game_time += dt
                
                # 处理事件
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.running = False
                    elif event.type == pygame.MOUSEBUTTONDOWN:
                        if event.button == 1:
                            self._handle_click(event.pos)
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            self.running = False
                        elif event.key == pygame.K_LEFT:
                            self._handle_click(self.buttons['left'].center)
                        elif event.key == pygame.K_RIGHT:
                            self._handle_click(self.buttons['right'].center)
                
                # 更新游戏状态
                self.audio.update(dt)
                self.camera.update(dt)
                self.particle_system.update(dt)
                self._update_terrain_from_audio()
                
                # 更新地形
                for row in self.terrain:
                    for block in row:
                        block.update_height(dt)
                
                # 更新角色
                self.character.update(dt, self.particle_system)
                
                # 渲染
                self._draw_gradient_background()
                
                camera_offset = self.camera.get_offset()
                
                # 绘制地形（从后往前）
                for z in range(len(self.terrain[0]) - 1, -1, -1):
                    for x in range(len(self.terrain)):
                        block = self.terrain[x][z]
                        if block.height > 0:
                            glow_color = block.get_glow_color(self.game_time)
                            self._draw_isometric_block_enhanced(
                                self.screen, block.x, block.z, 
                                block.height, block.color, camera_offset, glow_color
                            )
                            self._draw_decoration_enhanced(self.screen, block, camera_offset)
                
                # 绘制角色
                self._draw_character_enhanced(self.screen, camera_offset)
                
                # 绘制粒子效果
                self.particle_system.draw(self.screen)
                
                # 绘制UI
                self._draw_enhanced_ui(self.screen)
                
                pygame.display.flip()
                
        finally:
            self.audio.stop()
            pygame.quit()


if __name__ == "__main__":
    game = EnhancedVisualizationGame()
    game.run()