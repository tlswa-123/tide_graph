import pygame
import numpy as np
import pyaudio
import moderngl as mgl
import pyrr
from scipy.fft import fft
import threading
import queue
import math
import time
from typing import List, Tuple

class AudioProcessor:
    """音频处理类，负责从麦克风获取声音并分析"""
    
    def __init__(self, chunk_size=1024, sample_rate=44100):
        self.chunk_size = chunk_size
        self.sample_rate = sample_rate
        self.audio_queue = queue.Queue()
        self.volume = 0.0
        self.dominant_freq = 0.0
        self.is_running = False
        
        # 初始化PyAudio
        self.p = pyaudio.PyAudio()
        
        # 获取默认输入设备
        self.stream = self.p.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size,
            stream_callback=self._audio_callback
        )
    
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """音频回调函数"""
        audio_data = np.frombuffer(in_data, dtype=np.float32)
        if not self.audio_queue.full():
            self.audio_queue.put(audio_data)
        return (None, pyaudio.paContinue)
    
    def start(self):
        """开始音频处理"""
        self.is_running = True
        self.stream.start_stream()
        threading.Thread(target=self._process_audio, daemon=True).start()
    
    def stop(self):
        """停止音频处理"""
        self.is_running = False
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()
    
    def _process_audio(self):
        """处理音频数据"""
        while self.is_running:
            try:
                if not self.audio_queue.empty():
                    audio_data = self.audio_queue.get_nowait()
                    
                    # 计算音量
                    self.volume = np.sqrt(np.mean(audio_data**2))
                    
                    # 进行FFT分析获取主要频率
                    fft_data = np.abs(fft(audio_data))
                    freqs = np.fft.fftfreq(len(fft_data), 1/self.sample_rate)
                    
                    # 只考虑正频率部分
                    positive_freqs = freqs[:len(freqs)//2]
                    positive_fft = fft_data[:len(fft_data)//2]
                    
                    # 找到主要频率
                    if len(positive_fft) > 0:
                        peak_idx = np.argmax(positive_fft)
                        self.dominant_freq = abs(positive_freqs[peak_idx])
                
                time.sleep(0.01)  # 避免CPU占用过高
            except queue.Empty:
                continue


class Camera:
    """3D摄像机类"""
    
    def __init__(self, position=(10, 10, 10), target=(0, 0, 0)):
        self.position = np.array(position, dtype=np.float32)
        self.target = np.array(target, dtype=np.float32)
        self.up = np.array([0, 1, 0], dtype=np.float32)
        self.view_angle = 0  # 当前视角 (0=东南, 1=西南, 2=西北, 3=东北)
        self.transition_time = 0.5  # 视角切换时间
        self.current_transition = 0.0
        self.target_angle = 0
    
    def get_view_matrix(self):
        """获取视图矩阵"""
        return pyrr.matrix44.create_look_at(self.position, self.target, self.up)
    
    def update_view_angle(self, angle):
        """更新视角"""
        if angle != self.view_angle:
            self.target_angle = angle
            self.current_transition = 0.0
    
    def update(self, dt):
        """更新摄像机状态"""
        if self.current_transition < self.transition_time:
            self.current_transition = min(self.current_transition + dt, self.transition_time)
            
            # 插值计算当前位置
            progress = self.current_transition / self.transition_time
            # 使用easing函数使过渡更平滑
            progress = 1 - (1 - progress) ** 3
            
            # 计算目标位置
            angles = [
                (10, 10, 10),   # 东南
                (-10, 10, 10),  # 西南
                (-10, 10, -10), # 西北
                (10, 10, -10)   # 东北
            ]
            
            start_pos = np.array(angles[self.view_angle], dtype=np.float32)
            end_pos = np.array(angles[self.target_angle], dtype=np.float32)
            
            self.position = start_pos + (end_pos - start_pos) * progress
            
            if self.current_transition >= self.transition_time:
                self.view_angle = self.target_angle


class TerrainBlock:
    """地形方块类"""
    
    def __init__(self, x, z, height=0, decoration_type=0):
        self.x = x
        self.z = z
        self.height = height
        self.decoration_type = decoration_type  # 0=无, 1=树, 2=房子, 3=塔
        self.target_height = height
        self.color = self._get_color()
    
    def _get_color(self):
        """根据高度和装饰类型获取颜色"""
        base_colors = {
            0: (0.8, 0.9, 0.9),  # 浅蓝灰 (无装饰)
            1: (0.4, 0.8, 0.4),  # 绿色 (树)
            2: (0.9, 0.7, 0.5),  # 橙色 (房子)
            3: (0.7, 0.5, 0.9)   # 紫色 (塔)
        }
        return base_colors.get(self.decoration_type, (0.8, 0.8, 0.8))
    
    def update_from_audio(self, volume, frequency):
        """根据音频数据更新方块"""
        # 音量影响高度
        self.target_height = max(0.1, volume * 10)
        
        # 频率影响装饰类型
        if frequency < 200:
            self.decoration_type = 0  # 低频：无装饰
        elif frequency < 500:
            self.decoration_type = 1  # 中低频：树
        elif frequency < 1000:
            self.decoration_type = 2  # 中高频：房子
        else:
            self.decoration_type = 3  # 高频：塔
        
        self.color = self._get_color()
    
    def update_height(self, dt):
        """平滑更新高度"""
        if abs(self.height - self.target_height) > 0.01:
            self.height += (self.target_height - self.height) * dt * 5


class Character:
    """小人角色类"""
    
    def __init__(self, x=0, z=0):
        self.x = x
        self.z = z
        self.y = 0
        self.target_x = x
        self.target_z = z
        self.move_speed = 2.0
        self.path = []
        self.path_index = 0
    
    def set_target(self, x, z):
        """设置移动目标"""
        self.target_x = x
        self.target_z = z
    
    def update(self, dt, terrain):
        """更新角色位置"""
        # 移动到目标位置
        dx = self.target_x - self.x
        dz = self.target_z - self.z
        distance = math.sqrt(dx*dx + dz*dz)
        
        if distance > 0.1:
            move_dist = self.move_speed * dt
            if move_dist > distance:
                move_dist = distance
            
            self.x += (dx / distance) * move_dist
            self.z += (dz / distance) * move_dist
        
        # 更新Y坐标以站在地形上
        if terrain:
            grid_x = int(self.x + len(terrain) // 2)
            grid_z = int(self.z + len(terrain[0]) // 2)
            
            if 0 <= grid_x < len(terrain) and 0 <= grid_z < len(terrain[0]):
                self.y = terrain[grid_x][grid_z].height + 0.2


class SoundVisualizationGame:
    """主游戏类"""
    
    def __init__(self, width=1200, height=800):
        # 初始化Pygame
        pygame.init()
        self.screen = pygame.display.set_mode((width, height), pygame.OPENGL | pygame.DOUBLEBUF)
        pygame.display.set_caption("声音可视化游戏 - 纪念碑谷风格")
        
        # 初始化OpenGL
        self.ctx = mgl.create_context()
        self.ctx.enable(mgl.DEPTH_TEST)
        self.ctx.enable(mgl.CULL_FACE)
        
        # 游戏组件
        self.audio_processor = AudioProcessor()
        self.camera = Camera()
        self.terrain = self._create_terrain(20, 20)
        self.character = Character()
        
        # 游戏状态
        self.clock = pygame.time.Clock()
        self.running = True
        self.dt = 0
        
        # UI按钮
        self.buttons = self._create_buttons()
        
        # 着色器和渲染相关
        self._setup_rendering()
    
    def _create_terrain(self, width, height):
        """创建地形网格"""
        terrain = []
        for x in range(width):
            row = []
            for z in range(height):
                world_x = x - width // 2
                world_z = z - height // 2
                block = TerrainBlock(world_x, world_z)
                row.append(block)
            terrain.append(row)
        return terrain
    
    def _create_buttons(self):
        """创建UI按钮"""
        button_size = 60
        margin = 20
        screen_width, screen_height = pygame.display.get_surface().get_size()
        
        return {
            'left': pygame.Rect(screen_width - 2 * button_size - 2 * margin, 
                               screen_height - button_size - margin, 
                               button_size, button_size),
            'right': pygame.Rect(screen_width - button_size - margin, 
                                screen_height - button_size - margin, 
                                button_size, button_size)
        }
    
    def _setup_rendering(self):
        """设置渲染相关的着色器和缓冲区"""
        # 顶点着色器
        vertex_shader = '''
        #version 330 core
        
        in vec3 in_position;
        in vec3 in_color;
        
        uniform mat4 mvp;
        
        out vec3 v_color;
        
        void main() {
            gl_Position = mvp * vec4(in_position, 1.0);
            v_color = in_color;
        }
        '''
        
        # 片段着色器
        fragment_shader = '''
        #version 330 core
        
        in vec3 v_color;
        out vec4 fragColor;
        
        void main() {
            fragColor = vec4(v_color, 1.0);
        }
        '''
        
        self.program = self.ctx.program(vertex_shader=vertex_shader, 
                                       fragment_shader=fragment_shader)
    
    def _generate_cube_vertices(self, x, y, z, width, height, depth, color):
        """生成立方体顶点数据"""
        vertices = []
        
        # 定义立方体的8个顶点
        positions = [
            [x - width/2, y, z - depth/2],        # 0: 前左下
            [x + width/2, y, z - depth/2],        # 1: 前右下
            [x + width/2, y + height, z - depth/2], # 2: 前右上
            [x - width/2, y + height, z - depth/2], # 3: 前左上
            [x - width/2, y, z + depth/2],        # 4: 后左下
            [x + width/2, y, z + depth/2],        # 5: 后右下
            [x + width/2, y + height, z + depth/2], # 6: 后右上
            [x - width/2, y + height, z + depth/2]  # 7: 后左上
        ]
        
        # 定义立方体的6个面（三角形）
        faces = [
            # 前面
            [0, 1, 2], [0, 2, 3],
            # 后面
            [4, 7, 6], [4, 6, 5],
            # 左面
            [0, 3, 7], [0, 7, 4],
            # 右面
            [1, 5, 6], [1, 6, 2],
            # 顶面
            [3, 2, 6], [3, 6, 7],
            # 底面
            [0, 4, 5], [0, 5, 1]
        ]
        
        for face in faces:
            for vertex_idx in face:
                pos = positions[vertex_idx]
                vertices.extend(pos + list(color))
        
        return vertices
    
    def _render_terrain(self):
        """渲染地形"""
        vertices = []
        
        for row in self.terrain:
            for block in row:
                if block.height > 0:
                    cube_vertices = self._generate_cube_vertices(
                        block.x, 0, block.z, 0.9, block.height, 0.9, block.color
                    )
                    vertices.extend(cube_vertices)
                    
                    # 添加装饰物
                    if block.decoration_type > 0:
                        decoration_vertices = self._generate_decoration(block)
                        vertices.extend(decoration_vertices)
        
        if vertices:
            # 创建顶点缓冲区
            vertex_data = np.array(vertices, dtype=np.float32)
            vbo = self.ctx.buffer(vertex_data.tobytes())
            vao = self.ctx.vertex_array(self.program, [(vbo, '3f 3f', 'in_position', 'in_color')])
            
            # 设置MVP矩阵
            projection = pyrr.matrix44.create_perspective_projection_matrix(
                45, 1200/800, 0.1, 100.0
            )
            view = self.camera.get_view_matrix()
            model = pyrr.matrix44.create_identity()
            mvp = projection @ view @ model
            
            self.program['mvp'].write(mvp.astype(np.float32).tobytes())
            
            # 渲染
            vao.render()
    
    def _generate_decoration(self, block):
        """生成装饰物顶点"""
        vertices = []
        
        if block.decoration_type == 1:  # 树
            # 树干
            trunk_vertices = self._generate_cube_vertices(
                block.x, block.height, block.z, 0.2, 0.5, 0.2, (0.4, 0.2, 0.1)
            )
            vertices.extend(trunk_vertices)
            
            # 树冠
            crown_vertices = self._generate_cube_vertices(
                block.x, block.height + 0.5, block.z, 0.6, 0.4, 0.6, (0.2, 0.6, 0.2)
            )
            vertices.extend(crown_vertices)
            
        elif block.decoration_type == 2:  # 房子
            house_vertices = self._generate_cube_vertices(
                block.x, block.height, block.z, 0.6, 0.6, 0.6, (0.8, 0.6, 0.4)
            )
            vertices.extend(house_vertices)
            
            # 屋顶
            roof_vertices = self._generate_cube_vertices(
                block.x, block.height + 0.6, block.z, 0.8, 0.2, 0.8, (0.6, 0.3, 0.1)
            )
            vertices.extend(roof_vertices)
            
        elif block.decoration_type == 3:  # 塔
            for i in range(3):
                tower_vertices = self._generate_cube_vertices(
                    block.x, block.height + i * 0.4, block.z, 
                    0.4 - i * 0.05, 0.4, 0.4 - i * 0.05, 
                    (0.7 + i * 0.1, 0.5, 0.9 - i * 0.1)
                )
                vertices.extend(tower_vertices)
        
        return vertices
    
    def _render_character(self):
        """渲染角色"""
        # 身体
        body_vertices = self._generate_cube_vertices(
            self.character.x, self.character.y, self.character.z, 
            0.3, 0.6, 0.2, (0.9, 0.8, 0.7)
        )
        
        # 头部
        head_vertices = self._generate_cube_vertices(
            self.character.x, self.character.y + 0.6, self.character.z, 
            0.25, 0.25, 0.2, (0.95, 0.85, 0.75)
        )
        
        vertices = body_vertices + head_vertices
        
        if vertices:
            vertex_data = np.array(vertices, dtype=np.float32)
            vbo = self.ctx.buffer(vertex_data.tobytes())
            vao = self.ctx.vertex_array(self.program, [(vbo, '3f 3f', 'in_position', 'in_color')])
            
            # 设置MVP矩阵
            projection = pyrr.matrix44.create_perspective_projection_matrix(
                45, 1200/800, 0.1, 100.0
            )
            view = self.camera.get_view_matrix()
            model = pyrr.matrix44.create_identity()
            mvp = projection @ view @ model
            
            self.program['mvp'].write(mvp.astype(np.float32).tobytes())
            
            vao.render()
    
    def _render_ui(self):
        """渲染UI元素"""
        # 切换到2D渲染模式
        self.ctx.disable(mgl.DEPTH_TEST)
        
        # 这里可以添加2D UI渲染代码
        # 由于moderngl主要用于3D渲染，UI部分可以用pygame的2D功能
        
        self.ctx.enable(mgl.DEPTH_TEST)
    
    def _handle_click(self, pos):
        """处理点击事件"""
        if self.buttons['left'].collidepoint(pos):
            new_angle = (self.camera.view_angle - 1) % 4
            self.camera.update_view_angle(new_angle)
        elif self.buttons['right'].collidepoint(pos):
            new_angle = (self.camera.view_angle + 1) % 4
            self.camera.update_view_angle(new_angle)
    
    def _update_terrain_from_audio(self):
        """根据音频更新地形"""
        volume = self.audio_processor.volume
        frequency = self.audio_processor.dominant_freq
        
        # 更新地形中心区域的方块
        center_x = len(self.terrain) // 2
        center_z = len(self.terrain[0]) // 2
        
        # 影响半径随音量变化
        radius = max(1, int(volume * 5))
        
        for dx in range(-radius, radius + 1):
            for dz in range(-radius, radius + 1):
                x = center_x + dx
                z = center_z + dz
                
                if 0 <= x < len(self.terrain) and 0 <= z < len(self.terrain[0]):
                    distance = math.sqrt(dx*dx + dz*dz)
                    if distance <= radius:
                        # 距离越近，影响越大
                        influence = max(0, 1 - distance / radius)
                        adjusted_volume = volume * influence
                        
                        self.terrain[x][z].update_from_audio(adjusted_volume, frequency)
    
    def run(self):
        """主游戏循环"""
        self.audio_processor.start()
        
        try:
            while self.running:
                self.dt = self.clock.tick(60) / 1000.0
                
                # 处理事件
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.running = False
                    elif event.type == pygame.MOUSEBUTTONDOWN:
                        if event.button == 1:  # 左键点击
                            self._handle_click(event.pos)
                
                # 更新游戏状态
                self.camera.update(self.dt)
                self._update_terrain_from_audio()
                
                # 更新地形高度动画
                for row in self.terrain:
                    for block in row:
                        block.update_height(self.dt)
                
                # 更新角色
                self.character.update(self.dt, self.terrain)
                
                # 渲染
                self.ctx.clear(0.7, 0.9, 1.0)  # 天空蓝背景
                self._render_terrain()
                self._render_character()
                
                # 在这里可以添加2D UI渲染
                
                pygame.display.flip()
                
        finally:
            self.audio_processor.stop()
            pygame.quit()


if __name__ == "__main__":
    game = SoundVisualizationGame()
    game.run()