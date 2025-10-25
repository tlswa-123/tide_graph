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
    """海洋表面效果 - 完全波浪化的顶面"""
    
    def __init__(self):
        super().__init__()
        self.wave_offset = 0
        self.wave_grid_size = 12  # 波浪网格密度
    
    def update(self, dt):
        self.wave_offset += dt * 2
    
    def draw(self, screen, points, base_color, game_time):
        # 重新绘制波浪形状的整个表面
        if len(base_color) == 4 and len(points) >= 4:  # RGBA
            temp_surface = pygame.Surface((250, 250), pygame.SRCALPHA)
            min_x = min(p[0] for p in points)
            min_y = min(p[1] for p in points)
            
            # 计算菱形的中心和尺寸
            center_x = sum(p[0] for p in points) / 4 - min_x + 25
            center_y = sum(p[1] for p in points) / 4 - min_y + 25
            
            # 菱形的四个方向向量
            top_to_right = ((points[1][0] - points[0][0]) / 2, (points[1][1] - points[0][1]) / 2)
            top_to_left = ((points[3][0] - points[0][0]) / 2, (points[3][1] - points[0][1]) / 2)
            
            # 创建波浪网格
            wave_triangles = []
            grid_size = self.wave_grid_size
            
            for i in range(grid_size):
                for j in range(grid_size):
                    # 网格坐标 (0-1)
                    u1 = i / grid_size
                    v1 = j / grid_size
                    u2 = (i + 1) / grid_size
                    v2 = (j + 1) / grid_size
                    
                    # 转换为菱形坐标系
                    # 将正方形网格映射到菱形内部
                    if (u1 + v1 <= 1) and (u2 + v2 <= 1):  # 确保在菱形内
                        # 计算四个角点的位置
                        def get_wave_point(u, v):
                            # 菱形内的位置
                            base_x = center_x + top_to_right[0] * (2*u - 1) + top_to_left[0] * (2*v - 1)
                            base_y = center_y + top_to_right[1] * (2*u - 1) + top_to_left[1] * (2*v - 1)
                            
                            # 波浪高度
                            wave_height = (
                                math.sin(self.wave_offset + u * math.pi * 3) * 2 +
                                math.sin(self.wave_offset * 1.5 + v * math.pi * 2) * 1.5 +
                                math.sin(self.wave_offset * 0.8 + (u + v) * math.pi * 4) * 1
                            )
                            
                            return (base_x, base_y + wave_height)
                        
                        # 创建小三角形
                        p1 = get_wave_point(u1, v1)
                        p2 = get_wave_point(u2, v1)
                        p3 = get_wave_point(u1, v2)
                        p4 = get_wave_point(u2, v2)
                        
                        # 分割成两个三角形
                        wave_triangles.append([p1, p2, p3])
                        wave_triangles.append([p2, p4, p3])
            
            # 绘制所有波浪三角形
            for triangle in wave_triangles:
                if len(triangle) == 3:
                    # 根据波浪高度调整颜色
                    avg_y = sum(p[1] for p in triangle) / 3
                    brightness_factor = 0.8 + (center_y - avg_y) * 0.02  # 波峰更亮，波谷更暗
                    brightness_factor = max(0.6, min(1.2, brightness_factor))
                    
                    wave_color = tuple(
                        int(min(255, max(0, c * brightness_factor))) for c in base_color[:3]
                    ) + (base_color[3],)
                    
                    pygame.draw.polygon(temp_surface, wave_color, triangle)
            
            screen.blit(temp_surface, (min_x - 25, min_y - 25))
        else:
            # 备用简单渲染
            pygame.draw.polygon(screen, base_color, points)


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
        
        # 频率决定地形类型 - 根据实际音频数据调整范围
        if frequency < 150:  # 低频：海洋 (调整为更小范围)
            self.terrain_type = 0
            self.base_color = (64, 164, 223)
            self.terrain_name = "Ocean"
            self.surface_effect = OceanSurface()
        elif frequency < 300:  # 中频：沙漠 (150-300Hz)
            self.terrain_type = 1
            self.base_color = (255, 215, 0)  # 更鲜艳的金黄色
            self.terrain_name = "Desert"
            self.surface_effect = DesertSurface()
        else:  # 高频：草地 (>300Hz)
            self.terrain_type = 2
            self.base_color = (34, 139, 34)
            self.terrain_name = "Grassland"
            self.surface_effect = GrasslandSurface()
        
        # 音量决定透明度 - 重新调整范围让差异更明显
        # 根据实际数据：音量在0.0039-0.0047之间
        # 调整映射让这个小范围能产生大的透明度差异
        normalized_volume = (volume - 0.003) / (0.006 - 0.003)  # 将0.003-0.006映射到0-1
        alpha_factor = min(1.0, max(0.1, normalized_volume))  # 确保至少10%透明度
        self.alpha = int(255 * alpha_factor)
        
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
    
    def __init__(self, width=1200, height=800):
        pygame.init()
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("单方块声音可视化 - 地形表面效果")
        
        self.width = width
        self.height = height
        self.clock = pygame.time.Clock()
        self.running = True
        self.game_time = 0.0
        
        # 音频处理器
        self.audio_processor = RealAudioProcessor()
        
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
        """世界坐标转屏幕坐标 - 放大方块"""
        iso_x = (world_x - world_z) * 120  # 更大的等轴测间距
        iso_y = (world_x + world_z) * 60 - height  # 更大的等轴测间距
        
        screen_x = iso_x + self.width // 2 + camera_offset[0]
        screen_y = iso_y + self.height // 2 + camera_offset[1]
        
        return (screen_x, screen_y)
    
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
        """检查位置是否在屏幕边界内"""
        # 将世界坐标转换为屏幕坐标
        camera_offset = self.get_camera_offset()
        screen_x, screen_y = self.world_to_screen(x, z, 50, camera_offset)  # 使用中等高度检查
        
        # 留出边界 margin
        margin = 200
        return (margin < screen_x < self.width - margin and 
                margin < screen_y < self.height - margin)

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
        """更新方块"""
        # 检查是否有新声音
        new_sound = self.audio_processor.get_new_sound()
        if new_sound:
            # 计算新方块的位置
            if len(self.blocks) == 0:
                # 第一个方块在原点
                x, z = 0, 0
            else:
                # 紧挨着上一个方块生成
                last_block = self.blocks[-1]
                x, z = self.find_adjacent_position(last_block.x, last_block.z)
            
            # 记录位置
            pos_key = (int(x), int(z))
            self.used_positions.add(pos_key)
            
            # 创建新方块
            new_block = TerrainBlock3D(
                x, z,
                new_sound['volume'],
                new_sound['frequency'], 
                new_sound['duration'],
                self.game_time
            )
            
            self.blocks.append(new_block)
            
            # 限制方块数量
            if len(self.blocks) > self.max_blocks:
                removed_block = self.blocks.pop(0)
                # 移除旧位置记录
                old_pos_key = (int(removed_block.x), int(removed_block.z))
                self.used_positions.discard(old_pos_key)
        
        # 更新所有方块
        for block in self.blocks:
            block.update(dt)
    
    def update_camera(self, dt):
        """更新摄像机"""
        if self.camera_transition < 1.0:
            self.camera_transition = min(self.camera_transition + dt * 3, 1.0)
            if self.camera_transition >= 1.0:
                self.camera_angle = self.target_angle
    

    
    def draw_ui(self, screen):
        """绘制增强的UI界面"""
        # 半透明背景
        ui_surface = pygame.Surface((self.width, 200), pygame.SRCALPHA)
        ui_surface.fill((0, 0, 0, 180))
        screen.blit(ui_surface, (0, 0))
        
        # 标题和生成逻辑说明
        title = self.title_font.render("Audio-Driven 3D Terrain Generator", True, (255, 255, 255))
        screen.blit(title, (20, 10))
        
        # 生成逻辑说明（英文）
        logic_text = "Block Generation Logic: Duration → Height | Volume → Color Intensity | Frequency → Terrain Type"
        logic_surface = self.font.render(logic_text, True, (200, 255, 200))
        screen.blit(logic_surface, (20, 40))
        
        # 获取音频特征
        audio_features = self.audio_processor.get_audio_features()
        
        # 实时音频数据显示
        info_y = 70
        volume = audio_features['volume']
        frequency = audio_features['frequency']
        duration = audio_features['duration']
        
        # 音量显示和可视化条
        volume_text = f"Volume: {volume:.4f}"
        screen.blit(self.font.render(volume_text, True, (255, 255, 255)), (20, info_y))
        
        # 音量条
        volume_bar_x = 150
        volume_bar_width = 200
        volume_bar_height = 20
        pygame.draw.rect(screen, (60, 60, 60), (volume_bar_x, info_y, volume_bar_width, volume_bar_height))
        volume_fill = int(min(volume_bar_width, volume * volume_bar_width * 200))  # 转换为整数
        if volume_fill > 0:
            pygame.draw.rect(screen, (0, 255, 100), (volume_bar_x, info_y, volume_fill, volume_bar_height))
        pygame.draw.rect(screen, (200, 200, 200), (volume_bar_x, info_y, volume_bar_width, volume_bar_height), 2)
        
        # 频率显示和可视化条
        freq_y = info_y + 30
        freq_text = f"Frequency: {frequency:.0f}Hz"
        screen.blit(self.font.render(freq_text, True, (255, 255, 255)), (20, freq_y))
        
        # 频率条 (0-2000Hz范围)
        freq_bar_x = 150
        freq_bar_width = 200
        freq_bar_height = 20
        pygame.draw.rect(screen, (60, 60, 60), (freq_bar_x, freq_y, freq_bar_width, freq_bar_height))
        freq_fill = int(min(freq_bar_width, (frequency / 2000.0) * freq_bar_width))
        if freq_fill > 0:
            # 根据频率范围显示不同颜色
            if frequency < 200:
                freq_color = (64, 164, 223)  # 蓝色 (海洋)
            elif frequency < 600:
                freq_color = (238, 203, 173)  # 黄色 (沙漠)
            else:
                freq_color = (34, 139, 34)    # 绿色 (草地)
            pygame.draw.rect(screen, freq_color, (freq_bar_x, freq_y, freq_fill, freq_bar_height))
        pygame.draw.rect(screen, (200, 200, 200), (freq_bar_x, freq_y, freq_bar_width, freq_bar_height), 2)
        
        # 持续时间显示
        duration_y = freq_y + 30
        duration_text = f"Duration: {duration:.2f}s"
        screen.blit(self.font.render(duration_text, True, (255, 255, 255)), (20, duration_y))
        
        # 持续时间条
        duration_bar_x = 150
        duration_bar_width = 200
        duration_bar_height = 20
        pygame.draw.rect(screen, (60, 60, 60), (duration_bar_x, duration_y, duration_bar_width, duration_bar_height))
        duration_fill = int(min(duration_bar_width, (duration / 3.0) * duration_bar_width))  # 3秒为满
        if duration_fill > 0:
            pygame.draw.rect(screen, (255, 200, 0), (duration_bar_x, duration_y, duration_fill, duration_bar_height))
        pygame.draw.rect(screen, (200, 200, 200), (duration_bar_x, duration_y, duration_bar_width, duration_bar_height), 2)
        
        # 状态显示
        status_y = duration_y + 30
        status_color = (0, 255, 0) if audio_features['is_active'] else (255, 0, 0)
        if audio_features['waiting']:
            status_text = "Status: Waiting (5s cooldown)"
            status_color = (255, 255, 0)
        elif audio_features['is_active']:
            status_text = "Status: Detecting Sound - Generating Block!"
        else:
            status_text = "Status: Silent - Ready for Sound"
        
        status_surface = self.font.render(status_text, True, status_color)
        screen.blit(status_surface, (20, status_y))
        
        # 方块计数和最新方块信息
        block_count = len(self.blocks)
        count_text = f"Blocks: {block_count}/{self.max_blocks}"
        screen.blit(self.font.render(count_text, True, (255, 255, 255)), (400, info_y))
        
        # 最新方块信息
        if self.blocks:
            latest_block = self.blocks[-1]
            latest_info = f"Latest: {latest_block.terrain_name} H:{latest_block.target_height:.0f} V:{latest_block.original_volume:.4f}"
            latest_surface = self.font.render(latest_info, True, latest_block.color)
            screen.blit(latest_surface, (400, freq_y))
        
        # 地形类型说明
        terrain_info = "Terrain Types: Low Freq(<200Hz)→Ocean | Mid Freq(200-600Hz)→Desert | High Freq(>600Hz)→Grassland"
        terrain_surface = self.font.render(terrain_info, True, (180, 180, 180))
        screen.blit(terrain_surface, (20, status_y + 25))
        

        
        # 说明文字
        help_text = "每次声音生成一个方块 | 持续时间→高度 | 音量→颜色 | 频率→地形类型"
        help_surface = self.font.render(help_text, True, (255, 255, 255))
        help_rect = help_surface.get_rect()
        help_rect.centerx = self.width // 2
        help_rect.bottom = self.height - 10
        screen.blit(help_surface, help_rect)
    
    def run(self):
        """主游戏循环"""
        if not self.audio_processor.start():
            print("无法启动音频处理器!")
            return
        
        print("单方块游戏启动成功！每次声音只生成一个方块...")
        
        try:
            while self.running:
                dt = self.clock.tick(60) / 1000.0
                self.game_time += dt
                
                # 处理事件
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.running = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            self.running = False
                        elif event.key == pygame.K_c:
                            # 清除所有方块
                            self.blocks.clear()
                            print("清除所有方块")
                
                # 更新游戏状态
                self.update_camera(dt)
                self.update_blocks(dt)
                
                # 渲染
                # 渐变背景
                for y in range(self.height):
                    ratio = y / self.height
                    r = int(135 * (1 - ratio) + 100 * ratio)
                    g = int(206 * (1 - ratio) + 149 * ratio)
                    b = int(235 * (1 - ratio) + 237 * ratio)
                    pygame.draw.line(self.screen, (r, g, b), (0, y), (self.width, y))
                
                camera_offset = self.get_camera_offset()
                
                # 按距离排序绘制方块
                sorted_blocks = sorted(self.blocks, key=lambda b: b.x + b.z)
                
                for block in sorted_blocks:
                    if block.height > 0:
                        self.draw_3d_block(self.screen, block, camera_offset)
                
                # 绘制UI
                self.draw_ui(self.screen)
                
                pygame.display.flip()
                
        except KeyboardInterrupt:
            print("用户中断")
        finally:
            self.audio_processor.stop()
            pygame.quit()
            print("游戏结束")


if __name__ == "__main__":
    try:
        game = SingleBlockVisualizationGame()
        game.run()
    except Exception as e:
        print(f"游戏启动失败: {e}")
        import traceback
        traceback.print_exc()