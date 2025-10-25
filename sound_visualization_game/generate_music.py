#!/usr/bin/env python3
"""
简单的背景音乐生成器
生成轻松休闲的纯音乐作为游戏背景音乐
"""

import numpy as np
import pygame
import wave
import os

class BackgroundMusicGenerator:
    """背景音乐生成器"""
    
    def __init__(self, sample_rate=22050, duration=60):
        self.sample_rate = sample_rate
        self.duration = duration
        self.total_samples = int(sample_rate * duration)
        
    def generate_simple_melody(self):
        """生成简单的旋律"""
        # 轻松的大调音阶 (C大调)
        # C D E F G A B C (261, 294, 330, 349, 392, 440, 494, 523 Hz)
        notes = [261.63, 293.66, 329.63, 349.23, 392.00, 440.00, 493.88, 523.25]
        
        # 简单的旋律模式
        melody_pattern = [0, 2, 4, 2, 0, 4, 2, 0,  # C E G E C G E C
                         1, 3, 5, 3, 1, 5, 3, 1,  # D F A F D A F D  
                         2, 4, 6, 4, 2, 6, 4, 2,  # E G B G E B G E
                         0, 2, 4, 5, 4, 2, 0, 0]  # C E G A G E C C
        
        audio_data = np.zeros(self.total_samples)
        note_duration = 1.0  # 每个音符1秒
        samples_per_note = int(self.sample_rate * note_duration)
        
        for i, note_idx in enumerate(melody_pattern * 8):  # 重复8次
            start_sample = (i * samples_per_note) % self.total_samples
            end_sample = min(start_sample + samples_per_note, self.total_samples)
            
            if start_sample >= self.total_samples:
                break
                
            # 生成音符
            frequency = notes[note_idx]
            t = np.linspace(0, note_duration, end_sample - start_sample)
            
            # 主音调
            note = 0.3 * np.sin(2 * np.pi * frequency * t)
            # 添加泛音（更丰富的音色）
            note += 0.15 * np.sin(2 * np.pi * frequency * 2 * t)
            note += 0.08 * np.sin(2 * np.pi * frequency * 3 * t)
            
            # 包络（淡入淡出）
            envelope = np.exp(-t * 0.5) * (1 - np.exp(-t * 10))
            note *= envelope
            
            audio_data[start_sample:end_sample] += note
            
        return audio_data
    
    def generate_harmony(self, melody):
        """生成和声"""
        harmony = np.zeros_like(melody)
        
        # 简单的三度和声
        chord_pattern = [0, 0, 2, 2, 4, 4, 2, 0] * 16  # 和弦进行
        chord_freqs = [261.63, 329.63, 392.00]  # C E G
        
        samples_per_chord = len(melody) // len(chord_pattern)
        
        for i, chord_root in enumerate(chord_pattern):
            start_sample = i * samples_per_chord
            end_sample = min(start_sample + samples_per_chord, len(melody))
            
            if start_sample >= len(melody):
                break
                
            t = np.arange(end_sample - start_sample) / self.sample_rate
            
            # 生成三和弦
            for j, interval in enumerate([0, 2, 4]):  # 根音、三度、五度
                freq_idx = (chord_root + interval) % len(chord_freqs)
                frequency = chord_freqs[freq_idx] * 0.5  # 低八度
                
                chord_note = 0.1 * np.sin(2 * np.pi * frequency * t)
                harmony[start_sample:end_sample] += chord_note
                
        return harmony
    
    def add_ambient_effects(self, audio):
        """添加环境效果"""
        # 轻微的混响效果
        delay_samples = int(0.1 * self.sample_rate)  # 100ms延迟
        reverb = np.zeros_like(audio)
        
        if delay_samples < len(audio):
            reverb[delay_samples:] = 0.2 * audio[:-delay_samples]
            
        return audio + reverb
    
    def generate_music(self):
        """生成完整的背景音乐"""
        print("生成旋律...")
        melody = self.generate_simple_melody()
        
        print("生成和声...")
        harmony = self.generate_harmony(melody)
        
        print("混合音轨...")
        music = melody + harmony
        
        print("添加环境效果...")
        music = self.add_ambient_effects(music)
        
        # 标准化音量
        max_val = np.max(np.abs(music))
        if max_val > 0:
            music = music / max_val * 0.7  # 防止过响
        
        # 渐入渐出
        fade_samples = int(2 * self.sample_rate)  # 2秒渐变
        if len(music) > fade_samples * 2:
            # 渐入
            music[:fade_samples] *= np.linspace(0, 1, fade_samples)
            # 渐出
            music[-fade_samples:] *= np.linspace(1, 0, fade_samples)
        
        return music
    
    def save_to_wav(self, audio_data, filename):
        """保存为WAV文件"""
        # 转换为16位整数
        audio_16bit = (audio_data * 32767).astype(np.int16)
        
        with wave.open(filename, 'wb') as wav_file:
            wav_file.setnchannels(1)  # 单声道
            wav_file.setsampwidth(2)  # 16位
            wav_file.setframerate(self.sample_rate)
            wav_file.writeframes(audio_16bit.tobytes())

def generate_background_music():
    """生成背景音乐"""
    print("🎵 开始生成背景音乐...")
    
    generator = BackgroundMusicGenerator(duration=30)  # 30秒循环
    music = generator.generate_music()
    
    output_file = "background_music.wav"
    generator.save_to_wav(music, output_file)
    
    print(f"✅ 背景音乐已生成: {output_file}")
    print(f"   时长: {generator.duration}秒")
    print(f"   采样率: {generator.sample_rate}Hz")
    
    return output_file

if __name__ == "__main__":
    generate_background_music()