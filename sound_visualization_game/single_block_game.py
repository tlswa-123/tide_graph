"""
修改版真实音频控制的声音可视化游戏
每次声音只生成一个方块，带有地形表面效果
"""

import pygame
import numpy as np
import pyaudio
import math
import time
import queue
import threading
import os
from scipy import signal
from scipy.fft import fft
from typing import List, Tuple


class RealAudioProcessor:
    """真实音频处理类"""
    
    def __init__(self, chunk_size=1024, sample_rate=44100):
        self.chunk_size = chunk_size
        self.sample_rate = sample_rate
        self.audio_queue = queue.Queue(maxsize=10)
        
        # 音频特征
        self.volume = 0.0
        self.dominant_freq = 0.0
        self.sound_duration = 0.0
        self.is_sound_active = False
        self.last_sound_time = 0.0
        
        # 声音检测阈值
        self.volume_threshold = 0.005
        self.silence_duration = 5.0
        self.waiting_after_silence = False
        self.silence_start_time = 0.0
        
        # 音频缓冲区
        self.audio_buffer = np.zeros(sample_rate * 2, dtype=np.float32)
        
        # 新声音检测
        self.new_sound_detected = False
        
        # PyAudio设置
        self.p = None
        self.stream = None
        self.setup_audio()
    
    def find_input_device(self):
        """查找可用的输入设备"""
        if not self.p:
            return None
        
        try:
            default_input = self.p.get_default_input_device_info()
            device_index = default_input['index']
            print(f"使用默认麦克风设备: {default_input['name']}")
            return device_index
        except:
            print("默认设备不可用，寻找其他麦克风设备...")
            for i in range(self.p.get_device_count()):
                device_info = self.p.get_device_info_by_index(i)
                if (device_info['maxInputChannels'] > 0 and 
                    'mic' in device_info['name'].lower()):
                    print(f"使用麦克风设备: {device_info['name']}")
                    return i
            
            for i in range(self.p.get_device_count()):
                device_info = self.p.get_device_info_by_index(i)
                if device_info['maxInputChannels'] > 0:
                    print(f"使用输入设备: {device_info['name']}")
                    return i
            return None
    
    def setup_audio(self):
        """初始化音频系统"""
        try:
            self.p = pyaudio.PyAudio()
            input_device = self.find_input_device()
            
            if input_device is None:
                print("错误：没有找到可用的音频输入设备！")
                return False
            
            self.stream = self.p.open(
                format=pyaudio.paFloat32,
                channels=1,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size,
                input_device_index=input_device,
                stream_callback=self._audio_callback
            )
            
            print("音频流初始化成功")
            return True
            
        except Exception as e:
            print(f"音频初始化失败: {e}")
            return False
    
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """音频回调函数"""
        if status:
            print(f"音频状态: {status}")
        
        try:
            audio_data = np.frombuffer(in_data, dtype=np.float32)
            self.audio_queue.put(audio_data, block=False)
        except queue.Full:
            pass
        
        return (None, pyaudio.paContinue)
    
    def start(self):
        """开始音频处理"""
        if self.stream:
            self.stream.start_stream()
            threading.Thread(target=self._process_audio, daemon=True).start()
            print("音频处理开始")
            return True
        return False
    
    def stop(self):
        """停止音频处理"""
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        if self.p:
            self.p.terminate()
    
    def _process_audio(self):
        """处理音频数据的主循环"""
        while True:
            try:
                if not self.audio_queue.empty():
                    new_data = self.audio_queue.get_nowait()
                    
                    self.audio_buffer = np.roll(self.audio_buffer, -len(new_data))
                    self.audio_buffer[-len(new_data):] = new_data
                    
                    self._analyze_audio()
                
                time.sleep(0.01)
                
            except Exception as e:
                print(f"音频处理错误: {e}")
                break
    
    def _analyze_audio(self):
        """分析音频特征"""
        current_time = time.time()
        
        # 计算音量
        recent_buffer = self.audio_buffer[-self.sample_rate//4:]
        rms_volume = np.sqrt(np.mean(recent_buffer ** 2))
        self.volume = self.volume * 0.7 + rms_volume * 0.3
        
        # 检测声音是否活跃
        if self.volume > self.volume_threshold:
            if not self.is_sound_active:
                print(f"检测到新声音！音量: {self.volume:.3f}")
                self.is_sound_active = True
                self.sound_duration = 0.0
                self.waiting_after_silence = False
                # 不在这里立即标记新声音，而是在声音结束时
            
            self.sound_duration += 0.01
            self.last_sound_time = current_time
            
        else:
            if self.is_sound_active:
                print(f"声音结束，持续时间: {self.sound_duration:.2f}秒，音量峰值: {self.volume:.4f}")
                # 声音结束时标记有新声音，这样能捕获完整的时长
                self.new_sound_detected = True
                self.is_sound_active = False
                self.silence_start_time = current_time
                self.waiting_after_silence = True
        
        # 检查静默等待期
        if self.waiting_after_silence:
            if current_time - self.silence_start_time >= self.silence_duration:
                print("静默等待期结束，准备接收新声音")
                self.waiting_after_silence = False
        
        # FFT分析频率
        if self.is_sound_active and len(recent_buffer) > 0:
            self._analyze_frequency(recent_buffer)
    
    def _analyze_frequency(self, audio_data):
        """分析音频频率"""
        windowed = audio_data * np.hanning(len(audio_data))
        fft_data = np.abs(fft(windowed))
        freqs = np.fft.fftfreq(len(fft_data), 1/self.sample_rate)
        
        positive_freqs = freqs[:len(freqs)//2]
        positive_fft = fft_data[:len(fft_data)//2]
        
        if len(positive_fft) > 0:
            min_freq_idx = int(80 * len(positive_fft) / (self.sample_rate // 2))
            peak_idx = np.argmax(positive_fft[min_freq_idx:]) + min_freq_idx
            
            if peak_idx < len(positive_freqs):
                new_freq = abs(positive_freqs[peak_idx])
                self.dominant_freq = self.dominant_freq * 0.8 + new_freq * 0.2
    
    def get_new_sound(self):
        """获取新声音信息，只返回一次"""
        if self.new_sound_detected:
            self.new_sound_detected = False
            # 确保返回当前的音频特征
            return {
                'volume': self.volume,
                'frequency': self.dominant_freq,
                'duration': self.sound_duration
            }
        return None
    
    def get_audio_features(self):
        """获取当前音频特征"""
        return {
            'volume': self.volume,
            'frequency': self.dominant_freq,
            'duration': self.sound_duration,
            'is_active': self.is_sound_active,
            'waiting': self.waiting_after_silence
        }


class SurfaceEffect:
    """表面效果基类"""
    
    def __init__(self):
        self.time_offset = np.random.random() * math.pi * 2
    
    def update(self, dt):
        pass
    
    def draw(self, screen, points, base_color, game_time):
        # 支持透明度渲染
        if len(base_color) == 4:  # RGBA
            temp_surface = pygame.Surface((200, 200), pygame.SRCALPHA)
            pygame.draw.polygon(temp_surface, base_color, [(p[0] - min(pt[0] for pt in points) + 10, p[1] - min(pt[1] for pt in points) + 10) for p in points])
            screen.blit(temp_surface, (min(p[0] for p in points) - 10, min(p[1] for p in points) - 10))
        else:
            pygame.draw.polygon(screen, base_color, points)


class OceanSurface(SurfaceEffect):
    """海洋表面效果 - 动态波浪边框的顶面"""
    
    def __init__(self):
        super().__init__()
        self.wave_offset = 0
        self.wave_frequency = 3.0  # 波浪频率
        self.wave_amplitude = 4.0  # 波浪幅度
    
    def update(self, dt):
        self.wave_offset += dt * 2.5  # 波浪动画速度
    
    def create_wave_edge(self, start_point, end_point, segments=15):
        """创建两点间的波浪线"""
        wave_points = []
        for i in range(segments + 1):
            t = i / segments
            # 线性插值基础位置
            x = start_point[0] + (end_point[0] - start_point[0]) * t
            y = start_point[1] + (end_point[1] - start_point[1]) * t
            
            # 添加波浪偏移
            wave_phase = t * self.wave_frequency * math.pi * 2 + self.wave_offset
            wave_offset = math.sin(wave_phase) * self.wave_amplitude
            
            # 计算垂直方向 (相对于边的方向)
            edge_dx = end_point[0] - start_point[0] 
            edge_dy = end_point[1] - start_point[1]
            edge_length = math.sqrt(edge_dx**2 + edge_dy**2)
            
            if edge_length > 0:
                # 垂直向量 (顺时针旋转90度)
                normal_x = -edge_dy / edge_length
                normal_y = edge_dx / edge_length
                
                # 应用波浪偏移
                x += normal_x * wave_offset
                y += normal_y * wave_offset
            
            wave_points.append((int(x), int(y)))
        
        return wave_points
    
    def draw(self, screen, points, base_color, game_time):
        if len(points) >= 4:
            # 绘制填充的菱形（海洋表面）
            pygame.draw.polygon(screen, base_color[:3], points)
            
            # 绘制四条波浪边框
            wave_color = tuple(min(255, c + 40) for c in base_color[:3])  # 更亮的边框色
            
            # 四条边：top->right, right->bottom, bottom->left, left->top
            edges = [
                (points[0], points[1]),  # 上到右
                (points[1], points[2]),  # 右到下  
                (points[2], points[3]),  # 下到左
                (points[3], points[0])   # 左到上
            ]
            
            for start_pt, end_pt in edges:
                wave_points = self.create_wave_edge(start_pt, end_pt, segments=20)
                
                # 绘制波浪线
                if len(wave_points) > 1:
                    pygame.draw.lines(screen, wave_color, False, wave_points, 2)
                    
            # 添加一些水面反光效果
            self.draw_water_highlights(screen, points, base_color)
        else:
            # 备用简单渲染
            pygame.draw.polygon(screen, base_color, points)
    
    def draw_water_highlights(self, screen, points, base_color):
        """绘制水面反光效果"""
        if len(points) >= 4:
            # 在菱形内部添加一些闪烁的反光点
            center_x = sum(p[0] for p in points) / 4
            center_y = sum(p[1] for p in points) / 4
            
            # 添加几个闪烁的反光点
            import random
            random.seed(int(self.wave_offset * 10))  # 使用时间作为种子，创造闪烁效果
            
            for i in range(6):
                if random.random() > 0.3:  # 70%概率显示反光点
                    offset_x = random.uniform(-30, 30)
                    offset_y = random.uniform(-15, 15)
                    highlight_pos = (int(center_x + offset_x), int(center_y + offset_y))
                    
                    # 白色高光
                    highlight_color = (255, 255, 255, 180)
                    pygame.draw.circle(screen, highlight_color[:3], highlight_pos, 2)


class DesertSurface(SurfaceEffect):
    """沙漠表面效果 - 增强动态沙粒"""
    
    def __init__(self):
        super().__init__()
        self.sand_particles = []
        # 增加沙粒数量和大小
        for _ in range(40):  # 从15增加到40
            self.sand_particles.append({
                'x': np.random.random(),
                'y': np.random.random(),
                'speed': np.random.random() * 0.8 + 0.2,  # 速度更快
                'size': np.random.randint(2, 6),  # 更大的沙粒 (1-3 -> 2-6)
                'brightness': np.random.random() * 0.5 + 0.5  # 亮度变化
            })
    
    def update(self, dt):
        for particle in self.sand_particles:
            particle['x'] += particle['speed'] * dt * 0.3  # 速度稍快
            if particle['x'] > 1.1:  # 稍微超出边界再重置
                particle['x'] = -0.1
                particle['y'] = np.random.random()
                particle['brightness'] = np.random.random() * 0.5 + 0.5
    
    def draw(self, screen, points, base_color, game_time):
        # 支持透明度的沙漠表面
        if len(base_color) == 4:  # RGBA
            temp_surface = pygame.Surface((200, 200), pygame.SRCALPHA)
            min_x = min(p[0] for p in points)
            min_y = min(p[1] for p in points)
            temp_points = [(p[0] - min_x + 10, p[1] - min_y + 10) for p in points]
            pygame.draw.polygon(temp_surface, base_color, temp_points)
            
            # 绘制沙粒
            if len(temp_points) >= 4:
                t_min_x = min(p[0] for p in temp_points)
                t_max_x = max(p[0] for p in temp_points)
                t_min_y = min(p[1] for p in temp_points)
                t_max_y = max(p[1] for p in temp_points)
                
                # 绘制增强的沙粒效果
                for particle in self.sand_particles:
                    x = t_min_x + (t_max_x - t_min_x) * particle['x']
                    y = t_min_y + (t_max_y - t_min_y) * particle['y']
                    
                    # 根据亮度调整颜色
                    brightness = particle['brightness']
                    enhanced_color = tuple(
                        int(min(255, max(0, c * brightness))) for c in base_color[:3]
                    ) + (base_color[3],)
                    
                    # 绘制沙粒和小阴影
                    pygame.draw.circle(temp_surface, enhanced_color, (int(x), int(y)), particle['size'])
                    # 添加小阴影增强立体感
                    shadow_color = tuple(max(0, c - 50) for c in enhanced_color[:3]) + (enhanced_color[3]//2,)
                    pygame.draw.circle(temp_surface, shadow_color, (int(x+1), int(y+1)), max(1, particle['size']-1))
            
            screen.blit(temp_surface, (min_x - 10, min_y - 10))
        else:
            # 原始RGB渲染
            pygame.draw.polygon(screen, base_color, points)
            
            if len(points) >= 4:
                min_x = min(p[0] for p in points)
                max_x = max(p[0] for p in points)
                min_y = min(p[1] for p in points)
                max_y = max(p[1] for p in points)
                
                sand_color = tuple(max(0, c - 30) for c in base_color)
                
                for particle in self.sand_particles:
                    x = min_x + (max_x - min_x) * particle['x']
                    y = min_y + (max_y - min_y) * particle['y']
                    pygame.draw.circle(screen, sand_color, (int(x), int(y)), particle['size'])


class GrasslandSurface(SurfaceEffect):
    """草地表面效果 - 增强草坪纹理"""
    
    def __init__(self):
        super().__init__()
        self.grass_blades = []
        # 增加草叶数量和多样性
        for _ in range(60):  # 从20增加到60
            self.grass_blades.append({
                'x': np.random.random(),
                'y': np.random.random(),
                'height': np.random.randint(5, 15),  # 更高的草叶 (3-8 -> 5-15)
                'sway': np.random.random() * 2,
                'thickness': np.random.randint(2, 4),  # 草叶粗细
                'color_variant': np.random.randint(0, 3)  # 颜色变化
            })
    
    def update(self, dt):
        # 可以添加更复杂的草叶摆动
        pass
    
    def draw(self, screen, points, base_color, game_time):
        # 支持透明度的草地表面
        if len(base_color) == 4:  # RGBA
            temp_surface = pygame.Surface((200, 200), pygame.SRCALPHA)
            min_x = min(p[0] for p in points)
            min_y = min(p[1] for p in points)
            temp_points = [(p[0] - min_x + 10, p[1] - min_y + 10) for p in points]
            pygame.draw.polygon(temp_surface, base_color, temp_points)
            
            # 绘制增强的草叶效果
            if len(temp_points) >= 4:
                t_min_x = min(p[0] for p in temp_points)
                t_max_x = max(p[0] for p in temp_points)
                t_min_y = min(p[1] for p in temp_points)
                t_max_y = max(p[1] for p in temp_points)
                
                # 多种草叶颜色
                grass_colors = [
                    tuple(max(0, c - 30) for c in base_color[:3]) + (base_color[3],),  # 深绿
                    base_color,  # 原色
                    tuple(min(255, c + 30) for c in base_color[:3]) + (base_color[3],)  # 亮绿
                ]
                
                for blade in self.grass_blades:
                    x = t_min_x + (t_max_x - t_min_x) * blade['x']
                    y_base = t_min_y + (t_max_y - t_min_y) * blade['y']
                    
                    # 更强的摆动效果
                    sway = math.sin(game_time * 2.5 + blade['sway']) * 4
                    
                    color = grass_colors[blade['color_variant']]
                    start_pos = (int(x), int(y_base))
                    end_pos = (int(x + sway), int(y_base - blade['height']))
                    
                    # 更粗的草叶
                    pygame.draw.line(temp_surface, color, start_pos, end_pos, blade['thickness'])
                    
                    # 添加草叶顶端的小点
                    pygame.draw.circle(temp_surface, color, end_pos, max(1, blade['thickness']//2))
            
            screen.blit(temp_surface, (min_x - 10, min_y - 10))
        else:
            # 原始RGB渲染
            pygame.draw.polygon(screen, base_color, points)
            
            if len(points) >= 4:
                min_x = min(p[0] for p in points)
                max_x = max(p[0] for p in points)
                min_y = min(p[1] for p in points)
                max_y = max(p[1] for p in points)
                
                grass_colors = [
                    tuple(max(0, c - 20) for c in base_color),
                    tuple(min(255, c + 20) for c in base_color)
                ]
                
                for i, blade in enumerate(self.grass_blades):
                    x = min_x + (max_x - min_x) * blade['x']
                    y_base = min_y + (max_y - min_y) * blade['y']
                    
                    sway = math.sin(game_time * 2 + blade['sway']) * 2
                    
                    color = grass_colors[i % 2]
                    start_pos = (int(x), int(y_base))
                    end_pos = (int(x + sway), int(y_base - blade['height']))
                    
                    pygame.draw.line(screen, color, start_pos, end_pos, 2)


class TerrainBlock3D:
    """3D地形方块类 - 单个方块"""
    
    def __init__(self, x, z, volume, frequency, duration, creation_time):
        self.x = x
        self.z = z
        self.height = 0.0
        # 持续时间决定高度 - 增加范围和灵敏度
        self.target_height = max(20, min(200, duration * 150))
        self.creation_time = creation_time
        
        # 保存原始值用于调试
        self.original_volume = volume
        self.original_frequency = frequency
        self.original_duration = duration
        
        # 频率决定地形类型 - 降低草地阈值，让更多方块变成草地
        if frequency < 140:  # 低频：海洋
            self.terrain_type = 0
            self.base_color = (64, 164, 223)
            self.terrain_name = "Ocean"
            self.surface_effect = OceanSurface()
        elif frequency < 200:  # 中频：沙漠 (140-200Hz)
            self.terrain_type = 1
            self.base_color = (255, 215, 0)  # 更鲜艳的金黄色
            self.terrain_name = "Desert"
            self.surface_effect = DesertSurface()
        else:  # 高频：草地 (>200Hz) - 大幅降低阈值
            self.terrain_type = 2
            self.base_color = (34, 139, 34)
            self.terrain_name = "Grassland"
            self.surface_effect = GrasslandSurface()
        
        # 音量决定透明度 - 更广的范围和更平滑的过渡
        # 使用对数映射来处理音量的动态范围
        if volume <= 0.001:
            alpha_factor = 0.1  # 最低透明度
        elif volume >= 0.05:
            alpha_factor = 1.0  # 最高透明度
        else:
            # 对数映射，让小音量变化也有明显的透明度差异
            import math
            log_volume = math.log10(max(volume, 0.001))
            log_min = math.log10(0.001)  # -3
            log_max = math.log10(0.05)   # -1.3
            normalized = (log_volume - log_min) / (log_max - log_min)
            alpha_factor = 0.1 + 0.9 * normalized  # 0.1到1.0的范围
        
        self.alpha = int(255 * max(0.1, min(1.0, alpha_factor)))
        
        # 颜色保持原色，透明度单独控制
        self.color = self.base_color
        
        self.is_complete = False
        
        # 调试信息
        print(f"新方块: {self.terrain_name} | 高度: {self.target_height:.0f} | 音量: {volume:.4f} | 频率: {frequency:.0f}Hz | 时长: {duration:.2f}s | 透明度: {self.alpha}/255")
        
    def update(self, dt):
        """更新方块状态"""
        self.surface_effect.update(dt)
        
        # 平滑高度动画
        if abs(self.height - self.target_height) > 0.1:
            self.height += (self.target_height - self.height) * dt * 4
        else:
            self.is_complete = True


class SingleBlockVisualizationGame:
    """单方块声音可视化游戏"""
    
    def __init__(self, width=1400, height=800):
        pygame.init()
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Monument Valley Platform Game")
        
        self.width = width
        self.height = height
        
        # UI布局常数
        self.sidebar_width = 200  # 侧边栏宽度
        self.game_area_x = self.sidebar_width  # 游戏区域X起始位置
        self.game_area_width = self.width - 2 * self.sidebar_width  # 游戏区域宽度
        self.game_area_height = self.height  # 游戏区域高度
        self.clock = pygame.time.Clock()
        self.running = True
        self.game_time = 0.0
        
        # 音频处理器
        self.audio_processor = RealAudioProcessor()
        
        # 初始化pygame mixer用于背景音乐
        pygame.mixer.init(frequency=22050, size=-16, channels=1, buffer=1024)
        self.load_background_music()
        
        # 方块列表
        self.blocks = []
        self.max_blocks = 15  # 最多保存15个方块
        self.used_positions = set()  # 记录已使用的位置
        
        # 摄像机
        self.camera_angle = 0
        self.camera_transition = 1.0
        self.target_angle = 0
        
        # UI
        self.font = pygame.font.Font(None, 28)
        self.title_font = pygame.font.Font(None, 42)
        
        # 游戏模式和状态
        self.game_mode = "platformer"  # "audio" 或 "platformer"
        self.game_state = "playing"  # "playing", "won", "lost"
        self.blocks_used = 0
        self.max_game_blocks = 15
        
        # 声音状态追踪
        self.recent_sound_detected = False  # 最近是否检测到声音
        self.sound_detection_timeout = 5.0  # 5秒内的声音算有效
        self.last_sound_time = 0.0  # 上次检测到声音的时间
        
        # "no sound" 提示系统
        self.no_sound_message_active = False
        self.no_sound_message_start_time = 0.0
        self.no_sound_message_duration = 3.0  # 显示3秒
        
        # 游戏结束消息系统
        self.game_end_message_active = False
        self.game_end_message_start_time = 0.0
        self.game_end_message_duration = 5.0  # 显示5秒
        self.game_end_message_text = ""
        self.game_end_message_color = (255, 255, 255)
        
        # 初始化小人和终点（随机位置）
        import random
        self.player_start_x = random.randint(-3, 3)
        self.player_start_z = random.randint(-3, 3)
        self.player = Player(self.player_start_x, self.player_start_z)
        
        # 终点位置（确保不与小人重合）
        while True:
            goal_x = random.randint(-4, 4)
            goal_z = random.randint(-4, 4)
            distance = math.sqrt((goal_x - self.player_start_x)**2 + 
                               (goal_z - self.player_start_z)**2)
            if distance >= 3:  # 确保距离足够远
                break
        self.goal = Goal(goal_x, goal_z)
        
        # 等待第一个方块生成
        self.waiting_for_first_block = True
        self.first_block_generated = False
        
    def get_camera_offset(self):
        """获取摄像机偏移"""
        angles = [(120, 60), (-120, 60), (-120, -60), (120, -60)]
        
        if self.camera_transition < 1.0:
            start_offset = angles[self.camera_angle]
            end_offset = angles[self.target_angle]
            t = self.camera_transition
            smooth_t = t * t * (3 - 2 * t)
            offset_x = start_offset[0] + (end_offset[0] - start_offset[0]) * smooth_t
            offset_y = start_offset[1] + (end_offset[1] - start_offset[1]) * smooth_t
            return (offset_x, offset_y)
        
        return angles[self.camera_angle]
    
    def world_to_screen(self, world_x, world_z, height, camera_offset):
        """世界坐标转屏幕坐标 - 适应游戏区域"""
        iso_x = (world_x - world_z) * 120  # 更大的等轴测间距
        iso_y = (world_x + world_z) * 60 - height  # 更大的等轴测间距
        
        # 将游戏内容放在中间区域
        game_center_x = self.game_area_x + self.game_area_width // 2
        game_center_y = self.game_area_height // 2
        
        screen_x = iso_x + game_center_x + camera_offset[0]
        screen_y = iso_y + game_center_y + camera_offset[1]
        
        return (screen_x, screen_y)
    
    def screen_to_world(self, screen_x, screen_y):
        """将屏幕坐标转换为世界坐标（精确版）"""
        camera_offset = self.get_camera_offset()
        
        # 从游戏区域中心开始计算
        game_center_x = self.game_area_x + self.game_area_width // 2 + camera_offset[0]
        game_center_y = self.game_area_height // 2 + camera_offset[1]
        
        # 相对于游戏区域中心的偏移
        iso_x = screen_x - game_center_x
        iso_y = screen_y - game_center_y
        
        # 精确的逆等轴测变换
        # 原变换: iso_x = (world_x - world_z) * 120, iso_y = (world_x + world_z) * 60
        # 逆变换:
        # world_x - world_z = iso_x / 120
        # world_x + world_z = iso_y / 60
        # 解方程组:
        world_x = (iso_x / 120 + iso_y / 60) / 2
        world_z = (iso_y / 60 - iso_x / 120) / 2
        
        return (round(world_x), round(world_z))
    
    def draw_3d_block(self, screen, block, camera_offset):
        """绘制3D方块 - 支持透明度"""
        if block.height <= 0:
            return
        
        world_x, world_z = block.x, block.z
        height = block.height
        
        # 计算方块各个面的顶点
        base_x, base_y = self.world_to_screen(world_x, world_z, 0, camera_offset)
        top_x, top_y = self.world_to_screen(world_x, world_z, height, camera_offset)
        
        block_width = 120  # 更大的方块尺寸
        block_depth = 120
        
        # 顶面点
        top_points = [
            (top_x, top_y),
            (top_x + block_width//2, top_y + block_depth//4),
            (top_x, top_y + block_depth//2),
            (top_x - block_width//2, top_y + block_depth//4)
        ]
        
        # 创建带透明度的表面
        temp_surface = pygame.Surface((block_width * 2, int(height) + block_depth), pygame.SRCALPHA)
        
        # 相对坐标调整
        offset_x = block_width
        offset_y = 0
        
        # 左面 - 调整颜色和透明度
        left_color = tuple(max(0, c - 40) for c in block.color) + (block.alpha,)
        left_points_rel = [
            (offset_x - block_width//2, offset_y + block_depth//4),
            (offset_x - block_width//2, offset_y + int(height) + block_depth//4),
            (offset_x, offset_y + int(height) + block_depth//2),
            (offset_x, offset_y + block_depth//2)
        ]
        pygame.draw.polygon(temp_surface, left_color, left_points_rel)
        
        # 右面 - 调整颜色和透明度
        right_color = tuple(max(0, c - 60) for c in block.color) + (block.alpha,)
        right_points_rel = [
            (offset_x, offset_y + block_depth//2),
            (offset_x, offset_y + int(height) + block_depth//2),
            (offset_x + block_width//2, offset_y + int(height) + block_depth//4),
            (offset_x + block_width//2, offset_y + block_depth//4)
        ]
        pygame.draw.polygon(temp_surface, right_color, right_points_rel)
        
        # 将临时表面绘制到主屏幕
        screen.blit(temp_surface, (top_x - block_width, top_y))
        
        # 顶面 - 使用特殊表面效果（带透明度）
        block.surface_effect.draw(screen, top_points, (*block.color, block.alpha), self.game_time)
        
        # 灰色描边 - 绘制所有边框
        edge_color = (128, 128, 128)  # 灰色
        edge_width = 2
        
        # 顶面边框（菱形）
        pygame.draw.polygon(screen, edge_color, top_points, edge_width)
        
        # 左面边框
        left_points = [
            (top_x - block_width//2, top_y + block_depth//4),
            (top_x - block_width//2 + (top_x - top_x), top_y + block_depth//4 + int(height)),
            (top_x, top_y + block_depth//2 + int(height)),
            (top_x, top_y + block_depth//2)
        ]
        pygame.draw.lines(screen, edge_color, False, left_points, edge_width)
        
        # 右面边框
        right_points = [
            (top_x, top_y + block_depth//2),
            (top_x, top_y + block_depth//2 + int(height)),
            (top_x + block_width//2, top_y + block_depth//4 + int(height)),
            (top_x + block_width//2, top_y + block_depth//4)
        ]
        pygame.draw.lines(screen, edge_color, False, right_points, edge_width)
        
        # 垂直边线
        pygame.draw.line(screen, edge_color, 
                        (top_x - block_width//2, top_y + block_depth//4),
                        (top_x - block_width//2, top_y + block_depth//4 + int(height)), edge_width)
        pygame.draw.line(screen, edge_color,
                        (top_x + block_width//2, top_y + block_depth//4),
                        (top_x + block_width//2, top_y + block_depth//4 + int(height)), edge_width)
    
    def is_position_in_bounds(self, x, z):
        """检查位置是否在合理的游戏区域内"""
        # 简化的边界检查 - 基于世界坐标而不是屏幕坐标
        # 允许在小人周围较大的区域内放置方块
        max_distance = 10  # 最大距离
        return (abs(x) <= max_distance and abs(z) <= max_distance)

    def find_adjacent_position(self, base_x, base_z):
        """找到真正紧挨着且在边界内的位置"""
        import random
        
        # 所有8个方向，完全随机化
        all_directions = [
            (1, 0), (-1, 0), (0, 1), (0, -1),    # 四个主要方向
            (1, 1), (-1, 1), (1, -1), (-1, -1)   # 四个对角方向
        ]
        
        # 完全随机打乱所有方向
        random.shuffle(all_directions)
        
        # 优先尝试相邻位置
        for dx, dz in all_directions:
            new_x = base_x + dx
            new_z = base_z + dz
            pos_key = (int(new_x), int(new_z))
            
            # 检查是否在边界内且位置未被占用
            if (pos_key not in self.used_positions and 
                self.is_position_in_bounds(new_x, new_z)):
                return new_x, new_z
        
        # 如果相邻位置都不可用，尝试稍远一点的位置
        for distance in range(2, 5):
            # 生成该距离的所有可能位置
            candidates = []
            for dx in range(-distance, distance + 1):
                for dz in range(-distance, distance + 1):
                    if abs(dx) == distance or abs(dz) == distance:  # 只要边界上的点
                        candidates.append((dx, dz))
            
            random.shuffle(candidates)
            
            for dx, dz in candidates:
                new_x = base_x + dx
                new_z = base_z + dz
                pos_key = (int(new_x), int(new_z))
                
                if (pos_key not in self.used_positions and 
                    self.is_position_in_bounds(new_x, new_z)):
                    return new_x, new_z
        
        # 如果实在找不到合适位置，回到中心区域
        return 0, 0

    def update_blocks(self, dt):
        """更新方块 - 支持游戏模式"""
        if self.game_mode == "platformer":
            # 平台游戏模式：第一个方块通过声音生成，之后通过点击生成
            if self.waiting_for_first_block:
                new_sound = self.audio_processor.get_new_sound()
                if new_sound:
                    self.create_first_block(new_sound)
            
            # 检查是否有新声音（用于允许点击生成方块）
            new_sound = self.audio_processor.get_new_sound()
            if new_sound:
                # 更新声音状态
                self.recent_sound_detected = True
                self.last_sound_time = self.game_time
                print(f"🎵 声音已检测，现在可以点击放置方块！")
            
            # 检查声音是否过期
            if self.recent_sound_detected and (self.game_time - self.last_sound_time) > self.sound_detection_timeout:
                self.recent_sound_detected = False
                print("🔇 声音检测已过期，需要重新发出声音！")
        else:
            # 原来的音频模式
            new_sound = self.audio_processor.get_new_sound()
            if new_sound:
                if len(self.blocks) == 0:
                    x, z = 0, 0
                else:
                    last_block = self.blocks[-1]
                    x, z = self.find_adjacent_position(last_block.x, last_block.z)
                
                pos_key = (int(x), int(z))
                self.used_positions.add(pos_key)
                
                new_block = TerrainBlock3D(x, z, new_sound['volume'],
                                         new_sound['frequency'], 
                                         new_sound['duration'], self.game_time)
                self.blocks.append(new_block)
                
                if len(self.blocks) > self.max_blocks:
                    removed_block = self.blocks.pop(0)
                    old_pos_key = (int(removed_block.x), int(removed_block.z))
                    self.used_positions.discard(old_pos_key)
        
        # 更新所有方块
        for block in self.blocks:
            block.update(dt)
    
    def create_first_block(self, sound_data):
        """创建第一个方块在小人位置"""
        x, z = self.player_start_x, self.player_start_z
        pos_key = (int(x), int(z))
        self.used_positions.add(pos_key)
        
        new_block = TerrainBlock3D(x, z, sound_data['volume'],
                                 sound_data['frequency'], 
                                 sound_data['duration'], self.game_time)
        self.blocks.append(new_block)
        self.blocks_used += 1
        
        # 小人站在第一个方块上
        self.player.move_to_block(new_block)
        self.waiting_for_first_block = False
        self.first_block_generated = True
        print("第一个方块已生成在小人位置！")
    
    def create_block_at_position(self, x, z, sound_data=None):
        """在指定位置创建方块（点击生成）"""
        if self.blocks_used >= self.max_game_blocks:
            print("已达到最大方块数量！")
            return False
        
        # 确保使用整数坐标
        x, z = int(round(x)), int(round(z))
        pos_key = (x, z)
        
        if pos_key in self.used_positions:
            print(f"位置 ({x}, {z}) 已有方块！")
            return False
        
        if not self.is_position_in_bounds(x, z):
            print(f"位置 ({x}, {z}) 超出边界！")
            return False
        
        # 如果有声音数据，使用声音属性；否则使用默认属性
        if sound_data:
            new_block = TerrainBlock3D(x, z, sound_data['volume'],
                                     sound_data['frequency'], 
                                     sound_data['duration'], self.game_time)
            print(f"🎵 声音方块: {new_block.terrain_name} | 高度: {new_block.target_height:.0f} | 音量: {sound_data['volume']:.4f} | 频率: {sound_data['frequency']:.0f}Hz | 时长: {sound_data['duration']:.2f}s")
        else:
            # 获取当前音频特征作为方块属性
            audio_features = self.audio_processor.get_audio_features()
            new_block = TerrainBlock3D(x, z, audio_features['volume'],
                                     audio_features['frequency'], 
                                     audio_features['duration'], self.game_time)
            print(f"🎵 音频方块: {new_block.terrain_name} | 高度: {new_block.target_height:.0f} | 音量: {audio_features['volume']:.4f} | 频率: {audio_features['frequency']:.0f}Hz | 时长: {audio_features['duration']:.2f}s")
        
        self.blocks.append(new_block)
        self.blocks_used += 1
        self.used_positions.add(pos_key)
        
        print(f"✅ 方块已创建在位置: ({x}, {z}), 剩余: {self.max_game_blocks - self.blocks_used}")
        return True
    
    def handle_mouse_click(self, screen_pos):
        """处理鼠标点击事件"""
        if self.game_state != "playing":
            return
        
        if self.game_mode == "platformer" and self.first_block_generated:
            # 检查是否最近有声音
            if not self.recent_sound_detected:
                # 显示"no sound"提示
                self.show_no_sound_message()
                print("❌ 没有检测到声音！请先发出声音，然后点击放置方块。")
                return
            
            # 转换屏幕坐标到世界坐标
            world_x, world_z = self.screen_to_world(screen_pos[0], screen_pos[1])
            print(f"点击位置: 屏幕({screen_pos[0]}, {screen_pos[1]}) -> 世界({world_x}, {world_z})")
            
            # 创建方块
            if self.create_block_at_position(world_x, world_z):
                # 检查小人是否可以移动到新方块
                new_block = self.blocks[-1]
                if self.player.can_move_to_block(new_block, self.blocks):
                    # 自动移动到新方块
                    self.player.move_to_block(new_block)
                    print(f"小人自动移动到新方块 ({new_block.x}, {new_block.z})")
                
                # 消耗声音（一次声音只能生成一个方块）
                self.recent_sound_detected = False
                print("🎵 声音已消耗，需要重新发出声音才能继续放置方块！")
    
    def show_no_sound_message(self):
        """显示'no sound'提示"""
        self.no_sound_message_active = True
        self.no_sound_message_start_time = self.game_time
    
    def update_no_sound_message(self, dt):
        """更新'no sound'提示状态"""
        if self.no_sound_message_active:
            elapsed_time = self.game_time - self.no_sound_message_start_time
            if elapsed_time >= self.no_sound_message_duration:
                self.no_sound_message_active = False
    
    def show_game_end_message(self, text, color):
        """显示游戏结束提示"""
        self.game_end_message_active = True
        self.game_end_message_start_time = self.game_time
        self.game_end_message_text = text
        self.game_end_message_color = color
    
    def update_game_end_message(self, dt):
        """更新游戏结束提示状态"""
        if self.game_end_message_active:
            elapsed_time = self.game_time - self.game_end_message_start_time
            if elapsed_time >= self.game_end_message_duration:
                self.game_end_message_active = False
    
    def draw_no_sound_message(self, screen):
        """绘制'no sound'提示"""
        if self.no_sound_message_active:
            # 计算游戏区域中心
            center_x = self.game_area_x + self.game_area_width // 2
            center_y = self.game_area_height // 2
            
            # 创建半透明背景
            message_bg = pygame.Surface((300, 100), pygame.SRCALPHA)
            message_bg.fill((0, 0, 0, 180))
            
            # 绘制背景
            bg_rect = pygame.Rect(center_x - 150, center_y - 50, 300, 100)
            screen.blit(message_bg, bg_rect)
            
            # 绘制边框
            pygame.draw.rect(screen, (255, 100, 100), bg_rect, 3)
            
            # 绘制文字
            message_font = pygame.font.Font(None, 48)
            text_surface = message_font.render("NO SOUND", True, (255, 100, 100))
            text_rect = text_surface.get_rect(center=(center_x, center_y - 10))
            screen.blit(text_surface, text_rect)
            
            # 绘制提示文字
            hint_surface = self.font.render("Make some sound first!", True, (255, 255, 255))
            hint_rect = hint_surface.get_rect(center=(center_x, center_y + 20))
            screen.blit(hint_surface, hint_rect)
    
    def draw_game_end_message(self, screen):
        """绘制游戏结束提示"""
        if self.game_end_message_active:
            # 计算游戏区域中心
            center_x = self.game_area_x + self.game_area_width // 2
            center_y = self.game_area_height // 2
            
            # 创建半透明背景
            message_width = 400
            message_height = 120
            message_bg = pygame.Surface((message_width, message_height), pygame.SRCALPHA)
            message_bg.fill((0, 0, 0, 200))
            
            # 绘制背景
            bg_rect = pygame.Rect(center_x - message_width // 2, center_y - message_height // 2, 
                                 message_width, message_height)
            screen.blit(message_bg, bg_rect)
            
            # 绘制边框
            pygame.draw.rect(screen, self.game_end_message_color, bg_rect, 3)
            
            # 绘制消息文字
            big_font = pygame.font.Font(None, 48)
            text_surface = big_font.render(self.game_end_message_text, True, self.game_end_message_color)
            text_rect = text_surface.get_rect(center=(center_x, center_y - 15))
            screen.blit(text_surface, text_rect)
            
            # 添加重新开始提示
            restart_font = pygame.font.Font(None, 24)
            restart_text = "Press SPACE to restart or ESC to quit"
            restart_surface = restart_font.render(restart_text, True, (200, 200, 200))
            restart_rect = restart_surface.get_rect(center=(center_x, center_y + 25))
            screen.blit(restart_surface, restart_rect)
    
    def update_game_state(self, dt):
        """更新游戏状态"""
        if self.game_mode != "platformer":
            return
        
        # 更新小人和终点
        self.player.update(dt)
        self.goal.update(dt)
        
        # 检查胜利条件 - 需要几乎重合才算胜利
        if self.player.current_block and self.game_state == "playing":
            distance_to_goal = math.sqrt(
                (self.player.x - self.goal.x)**2 + 
                (self.player.z - self.goal.z)**2
            )
            if distance_to_goal <= 0.1:  # 必须几乎重合才算胜利
                self.game_state = "won"
                self.show_game_end_message("🎉 YOU WIN! 🎉", (255, 255, 0))
                print("🎉 游戏胜利！小人成功到达终点！🎉")
        
        # 检查失败条件
        if self.blocks_used >= self.max_game_blocks and self.game_state == "playing":
            distance_to_goal = math.sqrt(
                (self.player.x - self.goal.x)**2 + 
                (self.player.z - self.goal.z)**2
            )
            if distance_to_goal > 0.1:  # 调整失败判定距离
                self.game_state = "lost"
                self.show_game_end_message("💥 GAME OVER! 💥", (255, 100, 100))
                print("💥 游戏失败！方块用完了还没到达终点！")
    
    def handle_player_key_input(self, key):
        """处理玩家键盘输入"""
        if not self.player.current_block:
            return
            
        # 根据按键确定目标位置
        target_x = self.player.x
        target_z = self.player.z
        
        if key == pygame.K_w or key == pygame.K_UP:
            target_z -= 1
        elif key == pygame.K_s or key == pygame.K_DOWN:
            target_z += 1
        elif key == pygame.K_a or key == pygame.K_LEFT:
            target_x -= 1
        elif key == pygame.K_d or key == pygame.K_RIGHT:
            target_x += 1
        else:
            return  # 无效按键
        
        # 查找目标位置的方块，或者查找该方向上最近的方块
        target_block = None
        
        # 首先尝试找到精确位置的方块
        for block in self.blocks:
            if block.x == target_x and block.z == target_z and block.height > 0:
                target_block = block
                break
        
        # 如果没有精确位置的方块，查找该方向上最近的可到达方块
        if not target_block:
            direction_x = target_x - self.player.x
            direction_z = target_z - self.player.z
            
            best_block = None
            min_distance = float('inf')
            
            for block in self.blocks:
                if block.height > 0 and self.player.can_move_to_block(block, self.blocks):
                    # 检查方块是否在指定方向上
                    block_dir_x = block.x - self.player.x
                    block_dir_z = block.z - self.player.z
                    
                    # 检查方向是否大致相同（点积为正）
                    if (direction_x * block_dir_x + direction_z * block_dir_z) > 0:
                        distance = math.sqrt((block.x - self.player.x)**2 + (block.z - self.player.z)**2)
                        if distance < min_distance:
                            min_distance = distance
                            best_block = block
            
            target_block = best_block
        
        # 检查是否可以移动到目标方块
        if target_block and self.player.can_move_to_block(target_block, self.blocks):
            self.player.move_to_block(target_block)
            print(f"小人移动到 ({target_block.x}, {target_block.z})")
        else:
            print(f"无法移动到该方向 - 没有可用方块")
    
    def handle_player_movement(self):
        """处理小人移动（可以扩展为键盘控制）"""
        # 检查小人是否可以移动到相邻的方块
        for block in self.blocks:
            if block != self.player.current_block:
                if self.player.can_move_to_block(block, self.blocks):
                    # 这里可以添加键盘控制逻辑
                    pass
    
    def update_camera(self, dt):
        """更新摄像机"""
        if self.camera_transition < 1.0:
            self.camera_transition = min(self.camera_transition + dt * 3, 1.0)
            if self.camera_transition >= 1.0:
                self.camera_angle = self.target_angle
    

    
    def draw_ui(self, screen):
        """绘制分离的UI界面"""
        # 绘制左侧规则面板
        self.draw_left_panel(screen)
        
        # 绘制右侧信息面板
        self.draw_right_panel(screen)
        
        # 绘制游戏区域边界
        pygame.draw.line(screen, (100, 100, 100), 
                        (self.sidebar_width, 0), 
                        (self.sidebar_width, self.height), 2)
        pygame.draw.line(screen, (100, 100, 100), 
                        (self.width - self.sidebar_width, 0), 
                        (self.width - self.sidebar_width, self.height), 2)
    
    def draw_left_panel(self, screen):
        """绘制左侧游戏规则面板"""
        # 左侧背景
        left_bg = pygame.Surface((self.sidebar_width, self.height), pygame.SRCALPHA)
        left_bg.fill((0, 0, 0, 200))
        screen.blit(left_bg, (0, 0))
        
        y_offset = 20
        line_height = 25
        
        # 标题
        title = self.title_font.render("Rules", True, (255, 255, 100))
        screen.blit(title, (10, y_offset))
        y_offset += 40
        
        # 游戏规则
        rules = [
            "🎵 Make sound",
            "  generate 1st block",
            "",
            "🖱️ Click to place",
            "  more blocks",
            "",  
            "🚶 WASD to move",
            "  player",
            "",
            "🎯 Reach flag",
            "  to win!",
            "",
            "📊 Max 15 blocks",
            "",
            "⌨️ 'C' to restart"
        ]
        
        for rule in rules:
            if rule.strip():  # 非空行
                color = (200, 255, 200) if rule.startswith(('🎵', '🖱️', '🚶', '🎯')) else (255, 255, 255)
                text_surface = self.font.render(rule, True, color)
                screen.blit(text_surface, (10, y_offset))
            y_offset += line_height
    
    def draw_right_panel(self, screen):
        """绘制右侧实时信息面板"""
        # 右侧背景
        right_bg = pygame.Surface((self.sidebar_width, self.height), pygame.SRCALPHA)
        right_bg.fill((0, 0, 0, 200))
        right_x = self.width - self.sidebar_width
        screen.blit(right_bg, (right_x, 0))
        
        # 获取音频特征
        audio_features = self.audio_processor.get_audio_features()
        
        y_offset = 20
        line_height = 25
        
        # 标题
        title = self.title_font.render("Info", True, (255, 255, 100))
        screen.blit(title, (right_x + 10, y_offset))
        y_offset += 40
        
        # 实时音频信息和可视化条
        volume = audio_features['volume']
        frequency = audio_features['frequency']
        duration = audio_features['duration']
        
        # 音频标题
        audio_title = self.font.render("🎤 AUDIO DATA", True, (255, 255, 100))
        screen.blit(audio_title, (right_x + 10, y_offset))
        y_offset += 30
        
        # 绘制音量条
        self.draw_audio_bar(screen, right_x + 10, y_offset, "Volume", volume, 0.05, (0, 255, 100))
        y_offset += 35
        
        # 绘制频率条  
        self.draw_audio_bar(screen, right_x + 10, y_offset, "Frequency", frequency, 2000, (255, 200, 0))
        y_offset += 35
        
        # 绘制时长条
        self.draw_audio_bar(screen, right_x + 10, y_offset, "Duration", duration, 3.0, (100, 200, 255))
        y_offset += 45
        
        # 游戏状态
        status_title = self.font.render("📊 GAME STATUS", True, (255, 255, 100))
        screen.blit(status_title, (right_x + 10, y_offset))
        y_offset += 30
        
        blocks_text = f"Blocks: {self.blocks_used}/{self.max_game_blocks}"
        screen.blit(self.font.render(blocks_text, True, (255, 255, 255)), (right_x + 10, y_offset))
        y_offset += 30
        
        # 游戏状态显示
        if self.game_state == "won":
            status_text = "🎉 YOU WIN! 🎉"
            status_color = (255, 255, 0)
        elif self.game_state == "lost":
            status_text = "💥 GAME OVER"
            status_color = (255, 100, 100)
        elif self.waiting_for_first_block and not self.first_block_generated:
            status_text = "🔊 Make sound!"
            status_color = (100, 255, 255)
        else:
            status_text = "🎮 Playing..."
            status_color = (200, 255, 200)
            
        screen.blit(self.font.render(status_text, True, status_color), (right_x + 10, y_offset))
        y_offset += 35
            
        # 小人位置
        if hasattr(self, 'player') and self.player.current_block:
            player_title = self.font.render("🚶 PLAYER POS", True, (255, 255, 100))
            screen.blit(player_title, (right_x + 10, y_offset))
            y_offset += 25
            
            pos_text = f"X: {self.player.x}, Z: {self.player.z}"
            screen.blit(self.font.render(pos_text, True, (255, 255, 255)), (right_x + 10, y_offset))
    
    def draw_audio_bar(self, screen, x, y, label, value, max_value, color):
        """绘制音频可视化条"""
        bar_width = 160
        bar_height = 15
        
        # 标签和数值
        if label == "Volume":
            text = f"{label}: {value:.4f}"
        elif label == "Frequency":
            text = f"{label}: {value:.0f}Hz"
        else:  # Duration
            text = f"{label}: {value:.2f}s"
            
        label_surface = self.font.render(text, True, (255, 255, 255))
        screen.blit(label_surface, (x, y))
        
        # 进度条背景
        bar_y = y + 18
        pygame.draw.rect(screen, (60, 60, 60), (x, bar_y, bar_width, bar_height))
        
        # 进度条填充
        fill_ratio = min(1.0, value / max_value)
        fill_width = int(bar_width * fill_ratio)
        if fill_width > 0:
            pygame.draw.rect(screen, color, (x, bar_y, fill_width, bar_height))
        
        # 进度条边框
        pygame.draw.rect(screen, (200, 200, 200), (x, bar_y, bar_width, bar_height), 1)
    
    def load_background_music(self):
        """加载背景音乐"""
        music_file = "background_music.wav"
        if os.path.exists(music_file):
            try:
                pygame.mixer.music.load(music_file)
                pygame.mixer.music.set_volume(0.3)  # 设置较低音量，不干扰游戏
                print(f"🎵 背景音乐已加载: {music_file}")
            except pygame.error as e:
                print(f"❌ 背景音乐加载失败: {e}")
        else:
            print(f"⚠️  背景音乐文件不存在: {music_file}")
            print("   运行 generate_music.py 来生成背景音乐")
    
    def start_background_music(self):
        """开始播放背景音乐"""
        try:
            pygame.mixer.music.play(-1)  # -1表示无限循环
            print("🎵 背景音乐开始播放")
        except pygame.error as e:
            print(f"❌ 背景音乐播放失败: {e}")
    
    def stop_background_music(self):
        """停止背景音乐"""
        pygame.mixer.music.stop()
    
    def draw_enhanced_background(self):
        """绘制增强的渐变背景"""
        # 创建多层渐变背景
        # 主渐变：天空色调
        for y in range(self.height):
            ratio = y / self.height
            
            # 使用更自然的天空颜色
            if ratio < 0.3:  # 上层天空 - 浅蓝
                base_r, base_g, base_b = 173, 216, 230  # 浅天蓝
                target_r, target_g, target_b = 135, 206, 250  # 天蓝
                local_ratio = ratio / 0.3
            elif ratio < 0.7:  # 中层 - 蓝到橙
                base_r, base_g, base_b = 135, 206, 250  # 天蓝
                target_r, target_g, target_b = 255, 218, 185  # 桃色
                local_ratio = (ratio - 0.3) / 0.4
            else:  # 下层 - 温暖色调
                base_r, base_g, base_b = 255, 218, 185  # 桃色
                target_r, target_g, target_b = 255, 192, 203  # 粉红
                local_ratio = (ratio - 0.7) / 0.3
                
            r = int(base_r * (1 - local_ratio) + target_r * local_ratio)
            g = int(base_g * (1 - local_ratio) + target_g * local_ratio)
            b = int(base_b * (1 - local_ratio) + target_b * local_ratio)
            
            pygame.draw.line(self.screen, (r, g, b), (0, y), (self.width, y))
        
        # 添加动态云朵效果
        self.draw_background_clouds()
    
    def draw_background_clouds(self):
        """绘制背景云朵"""
        cloud_color = (255, 255, 255, 40)  # 半透明白色
        
        # 使用游戏时间创建缓慢移动的云朵
        cloud_positions = [
            (self.width * 0.2, self.height * 0.15, 60),
            (self.width * 0.7, self.height * 0.25, 80),
            (self.width * 0.4, self.height * 0.35, 50),
            (self.width * 0.85, self.height * 0.1, 70),
        ]
        
        for base_x, base_y, size in cloud_positions:
            # 云朵缓慢移动
            offset_x = (self.game_time * 5) % (self.width + 100) - 50
            cloud_x = (base_x + offset_x) % (self.width + 100) - 50
            
            # 绘制云朵（多个重叠圆形）
            if cloud_x > -100 and cloud_x < self.width + 100:
                # 创建半透明表面
                cloud_surface = pygame.Surface((size * 2, size), pygame.SRCALPHA)
                
                # 绘制多个重叠圆形组成云朵
                pygame.draw.circle(cloud_surface, cloud_color, (size//2, size//2), size//3)
                pygame.draw.circle(cloud_surface, cloud_color, (size//3, size//2), size//4)
                pygame.draw.circle(cloud_surface, cloud_color, (size*2//3, size//2), size//4)
                pygame.draw.circle(cloud_surface, cloud_color, (size//2, size//3), size//5)
                
                self.screen.blit(cloud_surface, (cloud_x - size, base_y - size//2))
    
    def restart_game(self):
        """重新开始游戏"""
        # 清除所有方块
        self.blocks.clear()
        self.used_positions.clear()
        self.blocks_used = 0
        
        # 重置游戏状态
        self.game_state = "playing"
        self.waiting_for_first_block = True
        self.first_block_generated = False
        
        # 重置声音状态
        self.recent_sound_detected = False
        self.last_sound_time = 0.0
        
        # 重置消息状态
        self.no_sound_message_active = False
        self.game_end_message_active = False
        
        # 重新随机化小人和终点位置
        import random
        self.player_start_x = random.randint(-3, 3)
        self.player_start_z = random.randint(-3, 3)
        self.player = Player(self.player_start_x, self.player_start_z)
        
        # 终点位置（确保不与小人重合）
        while True:
            goal_x = random.randint(-4, 4)
            goal_z = random.randint(-4, 4)
            distance = math.sqrt((goal_x - self.player_start_x)**2 + 
                               (goal_z - self.player_start_z)**2)
            if distance >= 3:  # 确保距离足够远
                break
        self.goal = Goal(goal_x, goal_z)
        
        print("🎮 游戏重新开始！")
    
    def run(self):
        """主游戏循环"""
        if not self.audio_processor.start():
            print("无法启动音频处理器!")
            return
        
        print("单方块游戏启动成功！每次声音只生成一个方块...")
        
        # 加载并播放背景音乐
        self.load_background_music()
        self.start_background_music()
        
        try:
            while self.running:
                dt = self.clock.tick(60) / 1000.0
                self.game_time += dt
                
                # 处理事件
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.running = False
                    elif event.type == pygame.MOUSEBUTTONDOWN:
                        if event.button == 1 and self.game_state == "playing":  # 只在游戏进行时处理点击
                            self.handle_mouse_click(event.pos)
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            self.running = False
                        elif event.key == pygame.K_SPACE and self.game_state in ["won", "lost"]:
                            # 游戏结束后按空格重新开始
                            self.restart_game()
                        elif event.key == pygame.K_c:
                            # 清除所有方块
                            self.blocks.clear()
                            self.used_positions.clear()
                            self.blocks_used = 0
                            self.waiting_for_first_block = True
                            self.first_block_generated = False
                            self.game_state = "playing"
                            # 重置小人和终点位置
                            if self.game_mode == "platformer":
                                self.player.x = 0
                                self.player.z = 0
                                self.player.current_block = None
                                self.goal.x = 10
                                self.goal.z = 10
                            print("游戏重置")
                        # 在平台游戏模式下添加玩家移动控制
                        elif self.game_mode == "platformer" and self.game_state == "playing":
                            self.handle_player_key_input(event.key)
                
                # 更新游戏状态
                self.update_camera(dt)
                self.update_blocks(dt)
                self.update_game_state(dt)
                
                # 渲染
                # 更美观的渐变背景
                self.draw_enhanced_background()
                
                camera_offset = self.get_camera_offset()
                
                # 按距离排序绘制方块
                sorted_blocks = sorted(self.blocks, key=lambda b: b.x + b.z)
                
                for block in sorted_blocks:
                    if block.height > 0:
                        self.draw_3d_block(self.screen, block, camera_offset)
                
                # 在平台游戏模式下绘制游戏元素
                if self.game_mode == "platformer":
                    # 更新小人和终点
                    self.player.update(dt)
                    self.goal.update(dt)
                    
                    # 绘制终点（在小人之前，这样小人会显示在上面）
                    self.goal.draw(self.screen, self)
                    
                    # 绘制小人
                    self.player.draw(self.screen, self)
                
                # 绘制UI
                self.draw_ui(self.screen)
                
                # 更新和绘制"no sound"提示
                self.update_no_sound_message(dt)
                self.draw_no_sound_message(self.screen)
                
                # 更新和绘制游戏结束消息
                self.update_game_end_message(dt)
                self.draw_game_end_message(self.screen)
                
                pygame.display.flip()
                
        except KeyboardInterrupt:
            print("用户中断")
        finally:
            self.audio_processor.stop()
            pygame.quit()
            print("游戏结束")


class Player:
    """游戏中的小人类"""
    
    def __init__(self, x, z):
        self.x = x  # 世界坐标X
        self.z = z  # 世界坐标Z
        self.current_block = None  # 当前站立的方块
        self.size = 15  # 小人大小
        self.color = (255, 100, 100)  # 红色小人
        self.animation_offset = 0  # 动画偏移
        
    def update(self, dt):
        """更新小人状态"""
        self.animation_offset += dt * 5  # 简单的上下跳动动画
    
    def can_move_to_block(self, target_block, current_blocks):
        """检查是否可以移动到目标方块"""
        if not target_block:
            return False
        
        # 如果没有当前方块，可以移动到任何方块
        if not self.current_block:
            return True
        
        # 检查目标方块是否与当前方块在合理距离内
        distance = math.sqrt((self.current_block.x - target_block.x)**2 + 
                           (self.current_block.z - target_block.z)**2)
        
        # 需要几乎重合才能移动 - 更严格的距离限制
        max_jump_distance = 1.1  # 减少到1.1个单位距离，几乎要重合
        return distance <= max_jump_distance
    
    def move_to_block(self, target_block):
        """移动到目标方块"""
        if target_block:
            self.x = target_block.x
            self.z = target_block.z
            self.current_block = target_block
            print(f"小人移动到方块: ({self.x}, {self.z})")
    
    def draw(self, screen, game):
        """绘制小人"""
        if self.current_block:
            # 计算小人在方块顶部的位置
            camera_offset = game.get_camera_offset()
            screen_x, screen_y = game.world_to_screen(self.x, self.z, 
                                                    self.current_block.height + 20, 
                                                    camera_offset)
            
            # 添加跳动动画
            bounce = math.sin(self.animation_offset) * 3
            screen_y += bounce
            
            # 绘制小人（简单的圆形）
            pygame.draw.circle(screen, self.color, (int(screen_x), int(screen_y)), self.size)
            pygame.draw.circle(screen, (0, 0, 0), (int(screen_x), int(screen_y)), self.size, 2)
            
            # 绘制简单的眼睛
            eye_offset = 5
            pygame.draw.circle(screen, (255, 255, 255), 
                             (int(screen_x - eye_offset), int(screen_y - 3)), 3)
            pygame.draw.circle(screen, (255, 255, 255), 
                             (int(screen_x + eye_offset), int(screen_y - 3)), 3)
            pygame.draw.circle(screen, (0, 0, 0), 
                             (int(screen_x - eye_offset), int(screen_y - 3)), 1)
            pygame.draw.circle(screen, (0, 0, 0), 
                             (int(screen_x + eye_offset), int(screen_y - 3)), 1)


class Goal:
    """终点旗子类"""
    
    def __init__(self, x, z):
        self.x = x
        self.z = z
        self.animation_offset = 0
        self.glow_size = 30
        
    def update(self, dt):
        """更新终点动画"""
        self.animation_offset += dt * 3
        
    def draw(self, screen, game):
        """绘制发光的终点旗子"""
        camera_offset = game.get_camera_offset()
        screen_x, screen_y = game.world_to_screen(self.x, self.z, 50, camera_offset)
        
        # 发光效果
        glow_alpha = int(100 + 50 * math.sin(self.animation_offset))
        glow_surface = pygame.Surface((self.glow_size * 2, self.glow_size * 2), pygame.SRCALPHA)
        
        # 多层发光
        for i in range(5):
            alpha = max(20, glow_alpha - i * 20)
            radius = self.glow_size - i * 3
            color = (255, 255, 0, alpha)  # 黄色发光
            pygame.draw.circle(glow_surface, color, 
                             (self.glow_size, self.glow_size), radius)
        
        screen.blit(glow_surface, 
                   (screen_x - self.glow_size, screen_y - self.glow_size))
        
        # 旗子杆
        pole_height = 40
        pygame.draw.line(screen, (139, 69, 19), 
                        (int(screen_x), int(screen_y)), 
                        (int(screen_x), int(screen_y - pole_height)), 4)
        
        # 旗子（三角形）
        flag_points = [
            (screen_x, screen_y - pole_height),
            (screen_x + 25, screen_y - pole_height + 8),
            (screen_x, screen_y - pole_height + 16)
        ]
        pygame.draw.polygon(screen, (255, 0, 0), flag_points)
        pygame.draw.polygon(screen, (0, 0, 0), flag_points, 2)


if __name__ == "__main__":
    try:
        game = SingleBlockVisualizationGame()
        game.run()
    except Exception as e:
        print(f"游戏启动失败: {e}")
        import traceback
        traceback.print_exc()