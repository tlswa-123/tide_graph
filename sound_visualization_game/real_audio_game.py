"""
真实音频控制的声音可视化游戏
基于pfad/week06的音频处理代码实现
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
    """真实音频处理类 - 基于pfad/week06的实现"""
    
    def __init__(self, chunk_size=1024, sample_rate=44100):
        self.chunk_size = chunk_size
        self.sample_rate = sample_rate
        self.audio_queue = queue.Queue(maxsize=10)
        
        # 音频特征
        self.volume = 0.0
        self.dominant_freq = 0.0
        self.sound_duration = 0.0  # 声音持续时间
        self.is_sound_active = False
        self.last_sound_time = 0.0
        
        # 声音检测阈值
        self.volume_threshold = 0.005  # 降低音量阈值，更容易检测到声音
        self.silence_duration = 5.0    # 静默等待时间（秒）
        self.waiting_after_silence = False
        self.silence_start_time = 0.0
        
        # 音频缓冲区
        self.audio_buffer = np.zeros(sample_rate * 2, dtype=np.float32)  # 2秒缓冲
        
        # PyAudio设置
        self.p = None
        self.stream = None
        self.setup_audio()
    
    def find_input_device(self):
        """查找可用的输入设备"""
        if not self.p:
            return None
        
        # 首先尝试使用默认输入设备
        try:
            default_input = self.p.get_default_input_device_info()
            device_index = default_input['index']
            print(f"使用默认麦克风设备: {default_input['name']}")
            return device_index
        except:
            # 如果默认设备不可用，查找其他设备
            print("默认设备不可用，寻找其他麦克风设备...")
            for i in range(self.p.get_device_count()):
                device_info = self.p.get_device_info_by_index(i)
                if (device_info['maxInputChannels'] > 0 and 
                    'mic' in device_info['name'].lower()):
                    print(f"使用麦克风设备: {device_info['name']}")
                    return i
            
            # 最后尝试任何输入设备
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
            pass  # 丢弃数据如果队列满了
        
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
                    
                    # 更新音频缓冲区
                    self.audio_buffer = np.roll(self.audio_buffer, -len(new_data))
                    self.audio_buffer[-len(new_data):] = new_data
                    
                    # 分析音频
                    self._analyze_audio()
                
                time.sleep(0.01)  # 避免过度占用CPU
                
            except Exception as e:
                print(f"音频处理错误: {e}")
                break
    
    def _analyze_audio(self):
        """分析音频特征"""
        current_time = time.time()
        
        # 计算音量（RMS）
        recent_buffer = self.audio_buffer[-self.sample_rate//4:]  # 最近0.25秒
        rms_volume = np.sqrt(np.mean(recent_buffer ** 2))
        
        # 平滑音量变化
        self.volume = self.volume * 0.7 + rms_volume * 0.3
        
        # 检测声音是否活跃
        if self.volume > self.volume_threshold:
            if not self.is_sound_active:
                print(f"检测到声音！音量: {self.volume:.3f}")
                self.is_sound_active = True
                self.sound_duration = 0.0
                self.waiting_after_silence = False
            
            self.sound_duration += 0.01  # 累计声音时长
            self.last_sound_time = current_time
            
        else:
            if self.is_sound_active:
                print(f"声音结束，持续时间: {self.sound_duration:.2f}秒")
                self.is_sound_active = False
                self.silence_start_time = current_time
                self.waiting_after_silence = True
        
        # 检查静默等待期
        if self.waiting_after_silence:
            if current_time - self.silence_start_time >= self.silence_duration:
                print("静默等待期结束，准备接收新声音")
                self.waiting_after_silence = False
        
        # FFT分析频率（仅在有声音时）
        if self.is_sound_active and len(recent_buffer) > 0:
            self._analyze_frequency(recent_buffer)
    
    def _analyze_frequency(self, audio_data):
        """分析音频频率"""
        # 应用窗口函数
        windowed = audio_data * np.hanning(len(audio_data))
        
        # FFT计算
        fft_data = np.abs(fft(windowed))
        freqs = np.fft.fftfreq(len(fft_data), 1/self.sample_rate)
        
        # 只考虑正频率部分
        positive_freqs = freqs[:len(freqs)//2]
        positive_fft = fft_data[:len(fft_data)//2]
        
        # 找到主要频率
        if len(positive_fft) > 0:
            # 忽略过低的频率（< 80Hz）
            min_freq_idx = int(80 * len(positive_fft) / (self.sample_rate // 2))
            peak_idx = np.argmax(positive_fft[min_freq_idx:]) + min_freq_idx
            
            if peak_idx < len(positive_freqs):
                new_freq = abs(positive_freqs[peak_idx])
                # 平滑频率变化
                self.dominant_freq = self.dominant_freq * 0.8 + new_freq * 0.2
    
    def can_generate_blocks(self):
        """检查是否可以生成方块"""
        return self.is_sound_active and not self.waiting_after_silence
    
    def get_audio_features(self):
        """获取音频特征"""
        return {
            'volume': self.volume,
            'frequency': self.dominant_freq,
            'duration': self.sound_duration,
            'is_active': self.is_sound_active,
            'waiting': self.waiting_after_silence
        }


class TerrainBlock3D:
    """3D地形方块类"""
    
    def __init__(self, x, z, terrain_type=0):
        self.x = x
        self.z = z
        self.height = 0.0
        self.target_height = 0.0
        self.terrain_type = terrain_type  # 0=海洋, 1=沙漠, 2=草地
        self.color = self._get_color()
        self.creation_time = time.time()
        self.is_generated = False
        
    def _get_color(self):
        """根据地形类型获取颜色"""
        terrain_colors = {
            0: (64, 164, 223),   # 海洋蓝
            1: (238, 203, 173),  # 沙漠黄
            2: (34, 139, 34)     # 草地绿
        }
        return terrain_colors.get(self.terrain_type, (128, 128, 128))
    
    def update_from_audio(self, volume, frequency, duration):
        """根据音频特征更新方块"""
        # 持续时间决定高度
        self.target_height = max(5, duration * 30)
        
        # 频率决定地形类型
        if frequency < 200:
            self.terrain_type = 0  # 低频 -> 海洋
        elif frequency < 600:
            self.terrain_type = 1  # 中频 -> 沙漠
        else:
            self.terrain_type = 2  # 高频 -> 草地
        
        # 音量决定颜色强度
        base_color = self._get_color()
        intensity = min(1.0, volume * 10)
        self.color = tuple(int(c * (0.5 + intensity * 0.5)) for c in base_color)
        
        self.is_generated = True
    
    def update_height(self, dt):
        """平滑更新高度"""
        if abs(self.height - self.target_height) > 0.1:
            self.height += (self.target_height - self.height) * dt * 3


class RealAudioVisualizationGame:
    """基于真实音频的可视化游戏"""
    
    def __init__(self, width=1400, height=900):
        pygame.init()
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("真实音频控制 - 3D地形生成器")
        
        self.width = width
        self.height = height
        self.clock = pygame.time.Clock()
        self.running = True
        self.game_time = 0.0
        
        # 音频处理器
        self.audio_processor = RealAudioProcessor()
        
        # 地形系统
        self.terrain_grid = {}  # 使用字典存储生成的方块
        self.current_generation_pos = [0, 0]  # 当前生成位置
        
        # 摄像机
        self.camera_angle = 0  # 0-3 对应四个视角
        self.camera_transition = 1.0
        self.target_angle = 0
        
        # UI
        self.font = pygame.font.Font(None, 28)
        self.title_font = pygame.font.Font(None, 42)
        
        # 按钮
        button_size = 60
        margin = 20
        self.buttons = {
            'left': pygame.Rect(width - 2*button_size - 2*margin, 
                               height - button_size - margin, 
                               button_size, button_size),
            'right': pygame.Rect(width - button_size - margin, 
                                height - button_size - margin, 
                                button_size, button_size)
        }
        
    def get_camera_offset(self):
        """获取摄像机偏移"""
        angles = [(150, 80), (-150, 80), (-150, -80), (150, -80)]
        
        if self.camera_transition < 1.0:
            start_offset = angles[self.camera_angle]
            end_offset = angles[self.target_angle]
            t = self.camera_transition
            smooth_t = t * t * (3 - 2 * t)  # 平滑插值
            offset_x = start_offset[0] + (end_offset[0] - start_offset[0]) * smooth_t
            offset_y = start_offset[1] + (end_offset[1] - start_offset[1]) * smooth_t
            return (offset_x, offset_y)
        
        return angles[self.camera_angle]
    
    def world_to_screen(self, world_x, world_z, height, camera_offset):
        """世界坐标转换为屏幕坐标（等轴测投影）"""
        # 等轴测投影
        iso_x = (world_x - world_z) * 30
        iso_y = (world_x + world_z) * 15 - height
        
        # 应用摄像机偏移
        screen_x = iso_x + self.width // 2 + camera_offset[0]
        screen_y = iso_y + self.height // 2 + camera_offset[1]
        
        return (screen_x, screen_y)
    
    def draw_3d_block(self, screen, world_x, world_z, height, color, camera_offset):
        """绘制3D方块"""
        if height <= 0:
            return
        
        # 计算方块的各个面
        base_x, base_y = self.world_to_screen(world_x, world_z, 0, camera_offset)
        top_x, top_y = self.world_to_screen(world_x, world_z, height, camera_offset)
        
        # 方块尺寸
        block_width = 30
        block_depth = 30
        
        # 顶面
        top_points = [
            (top_x, top_y),
            (top_x + block_width//2, top_y + block_depth//4),
            (top_x, top_y + block_depth//2),
            (top_x - block_width//2, top_y + block_depth//4)
        ]
        pygame.draw.polygon(screen, color, top_points)
        
        # 左面（暗一些）
        left_color = tuple(max(0, c - 40) for c in color)
        left_points = [
            (top_x - block_width//2, top_y + block_depth//4),
            (base_x - block_width//2, base_y + block_depth//4),
            (base_x, base_y + block_depth//2),
            (top_x, top_y + block_depth//2)
        ]
        pygame.draw.polygon(screen, left_color, left_points)
        
        # 右面（更暗）
        right_color = tuple(max(0, c - 60) for c in color)
        right_points = [
            (top_x, top_y + block_depth//2),
            (base_x, base_y + block_depth//2),
            (base_x + block_width//2, base_y + block_depth//4),
            (top_x + block_width//2, top_y + block_depth//4)
        ]
        pygame.draw.polygon(screen, right_color, right_points)
        
        # 绘制边框
        pygame.draw.polygon(screen, (0, 0, 0), top_points, 1)
        pygame.draw.polygon(screen, (0, 0, 0), left_points, 1)
        pygame.draw.polygon(screen, (0, 0, 0), right_points, 1)
    
    def update_terrain_generation(self):
        """更新地形生成"""
        audio_features = self.audio_processor.get_audio_features()
        
        if self.audio_processor.can_generate_blocks():
            # 生成新方块
            x, z = self.current_generation_pos
            key = (x, z)
            
            if key not in self.terrain_grid:
                self.terrain_grid[key] = TerrainBlock3D(x, z)
            
            # 更新方块属性
            self.terrain_grid[key].update_from_audio(
                audio_features['volume'],
                audio_features['frequency'],
                audio_features['duration']
            )
            
            # 移动到下一个位置（螺旋生成）
            self._move_generation_position()
    
    def _move_generation_position(self):
        """螺旋式移动生成位置"""
        x, z = self.current_generation_pos
        
        # 简单的螺旋逻辑
        if abs(x) <= abs(z) and (x != z or x >= 0):
            x += 1 if z >= 0 else -1
        else:
            z += 1 if x < 0 else -1
        
        self.current_generation_pos = [x, z]
    
    def update_camera(self, dt):
        """更新摄像机"""
        if self.camera_transition < 1.0:
            self.camera_transition = min(self.camera_transition + dt * 3, 1.0)
            if self.camera_transition >= 1.0:
                self.camera_angle = self.target_angle
    
    def handle_click(self, pos):
        """处理点击事件"""
        if self.buttons['left'].collidepoint(pos):
            self.target_angle = (self.camera_angle - 1) % 4
            self.camera_transition = 0.0
        elif self.buttons['right'].collidepoint(pos):
            self.target_angle = (self.camera_angle + 1) % 4
            self.camera_transition = 0.0
    
    def draw_ui(self, screen):
        """绘制UI"""
        # 半透明背景
        ui_surface = pygame.Surface((self.width, 120), pygame.SRCALPHA)
        ui_surface.fill((0, 0, 0, 150))
        screen.blit(ui_surface, (0, 0))
        
        # 标题
        title = self.title_font.render("真实音频3D地形生成器", True, (255, 255, 255))
        screen.blit(title, (20, 15))
        
        # 音频信息
        audio_features = self.audio_processor.get_audio_features()
        info_y = 55
        
        volume_text = f"音量: {audio_features['volume']:.3f}"
        freq_text = f"频率: {audio_features['frequency']:.0f}Hz"
        duration_text = f"持续: {audio_features['duration']:.1f}s"
        
        screen.blit(self.font.render(volume_text, True, (255, 255, 255)), (20, info_y))
        screen.blit(self.font.render(freq_text, True, (255, 255, 255)), (200, info_y))
        screen.blit(self.font.render(duration_text, True, (255, 255, 255)), (380, info_y))
        
        # 状态指示器
        status_color = (0, 255, 0) if audio_features['is_active'] else (255, 0, 0)
        if audio_features['waiting']:
            status_text = "等待中..."
            status_color = (255, 255, 0)
        elif audio_features['is_active']:
            status_text = "生成中!"
        else:
            status_text = "静默"
        
        status_surface = self.font.render(status_text, True, status_color)
        screen.blit(status_surface, (550, info_y))
        
        # 生成的方块数量
        block_count = len(self.terrain_grid)
        count_text = f"方块数: {block_count}"
        screen.blit(self.font.render(count_text, True, (255, 255, 255)), (20, info_y + 25))
        
        # 视角按钮
        for button_name, button_rect in self.buttons.items():
            pygame.draw.rect(screen, (70, 70, 70), button_rect)
            pygame.draw.rect(screen, (150, 150, 150), button_rect, 2)
            
            icon = "◀" if button_name == 'left' else "▶"
            icon_surface = self.font.render(icon, True, (255, 255, 255))
            icon_rect = icon_surface.get_rect(center=button_rect.center)
            screen.blit(icon_surface, icon_rect)
        
        # 说明文字
        view_names = ["东南视角", "西南视角", "西北视角", "东北视角"]
        view_text = f"当前: {view_names[self.camera_angle]}"
        screen.blit(self.font.render(view_text, True, (255, 255, 255)), (self.width - 200, 20))
        
        help_text = "对着麦克风发声，根据声音生成3D地形方块"
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
        
        print("游戏启动成功！开始对着麦克风说话...")
        
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
                            self.handle_click(event.pos)
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            self.running = False
                        elif event.key == pygame.K_LEFT:
                            self.handle_click(self.buttons['left'].center)
                        elif event.key == pygame.K_RIGHT:
                            self.handle_click(self.buttons['right'].center)
                
                # 更新游戏状态
                self.update_camera(dt)
                self.update_terrain_generation()
                
                # 更新所有方块的高度动画
                for block in self.terrain_grid.values():
                    block.update_height(dt)
                
                # 渲染
                self.screen.fill((135, 206, 235))  # 天空蓝背景
                
                camera_offset = self.get_camera_offset()
                
                # 绘制所有地形方块（按Z坐标排序以正确显示前后关系）
                sorted_blocks = sorted(self.terrain_grid.items(), 
                                     key=lambda item: item[0][1] + item[0][0])
                
                for (x, z), block in sorted_blocks:
                    if block.is_generated and block.height > 0:
                        self.draw_3d_block(self.screen, x, z, block.height, 
                                         block.color, camera_offset)
                
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
        game = RealAudioVisualizationGame()
        game.run()
    except Exception as e:
        print(f"游戏启动失败: {e}")
        import traceback
        traceback.print_exc()