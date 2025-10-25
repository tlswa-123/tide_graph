"""
音频设备测试工具
用于验证麦克风是否正常工作
"""

import pyaudio
import numpy as np
import time

def list_audio_devices():
    """列出所有音频设备"""
    p = pyaudio.PyAudio()
    print("=== 可用音频设备 ===")
    
    input_devices = []
    for i in range(p.get_device_count()):
        device_info = p.get_device_info_by_index(i)
        if device_info['maxInputChannels'] > 0:
            input_devices.append((i, device_info))
            print(f"输入设备 {i}: {device_info['name']}")
            print(f"  - 最大输入通道: {device_info['maxInputChannels']}")
            print(f"  - 采样率: {device_info['defaultSampleRate']}")
            print()
    
    p.terminate()
    return input_devices

def test_microphone(device_index=None):
    """测试麦克风输入"""
    p = pyaudio.PyAudio()
    
    # 音频参数
    chunk_size = 1024
    sample_rate = 44100
    
    try:
        if device_index is None:
            # 使用默认输入设备
            stream = p.open(
                format=pyaudio.paFloat32,
                channels=1,
                rate=sample_rate,
                input=True,
                frames_per_buffer=chunk_size
            )
            print("使用默认输入设备")
        else:
            stream = p.open(
                format=pyaudio.paFloat32,
                channels=1,
                rate=sample_rate,
                input=True,
                frames_per_buffer=chunk_size,
                input_device_index=device_index
            )
            print(f"使用设备 {device_index}")
        
        print("开始录音测试...")
        print("请对着麦克风说话或制造声音，观察音量变化")
        print("按 Ctrl+C 停止测试")
        print("-" * 50)
        
        try:
            while True:
                # 读取音频数据
                data = stream.read(chunk_size, exception_on_overflow=False)
                audio_array = np.frombuffer(data, dtype=np.float32)
                
                # 计算音量
                volume = np.sqrt(np.mean(audio_array ** 2))
                
                # 计算主要频率
                fft_data = np.abs(np.fft.fft(audio_array))
                freqs = np.fft.fftfreq(len(fft_data), 1/sample_rate)
                positive_freqs = freqs[:len(freqs)//2]
                positive_fft = fft_data[:len(fft_data)//2]
                
                if len(positive_fft) > 0:
                    peak_idx = np.argmax(positive_fft)
                    dominant_freq = abs(positive_freqs[peak_idx])
                else:
                    dominant_freq = 0
                
                # 显示结果
                volume_bar = "█" * int(volume * 100)
                print(f"\r音量: {volume:.4f} [{volume_bar:<20}] 频率: {dominant_freq:.0f}Hz", end="")
                
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            print("\n测试结束")
        
        stream.stop_stream()
        stream.close()
        
    except Exception as e:
        print(f"麦克风测试失败: {e}")
    finally:
        p.terminate()

def main():
    print("音频设备测试工具")
    print("=" * 30)
    
    # 列出设备
    devices = list_audio_devices()
    
    if not devices:
        print("未找到可用的输入设备！")
        return
    
    # 选择设备
    print("选择测试设备:")
    print("0: 使用默认设备")
    for i, (device_idx, device_info) in enumerate(devices):
        print(f"{i+1}: 设备 {device_idx} - {device_info['name']}")
    
    try:
        choice = input("\n请输入选择 (回车使用默认): ").strip()
        
        if choice == "" or choice == "0":
            test_microphone()
        else:
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(devices):
                device_idx, _ = devices[choice_idx]
                test_microphone(device_idx)
            else:
                print("无效选择")
                
    except ValueError:
        print("无效输入")
    except Exception as e:
        print(f"错误: {e}")

if __name__ == "__main__":
    main()