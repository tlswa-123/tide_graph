# 🎵 Sound Visualization Platform Game

A Monument Valley-style 3D platform game controlled by your voice! Build pathways with sound and guide your character to victory.

## 🎬 Live Game Demo

### 🎮 **Watch the Gameplay in Action!**

<div align="center">

### 🎬 **Live Gameplay Recording**

> **Demo Video:** [🎥 View Gameplay Recording](https://github.com/tlswa-123/sound_visualization_game/blob/main/game_screenshot.mp4)

https://github.com/user-attachments/assets/game_screenshot.mp4

**🎮 Watch the voice-controlled block building in action!**  
*Speak into your microphone → Generate 3D blocks → Guide character to victory!*

> 💡 **Click the video link above to see the full Monument Valley-style 3D platformer game in action!**

</div>

> � **How it works:** Speak into your microphone → Colorful 3D blocks appear → Guide your character to the golden flag!

---

## 🚀 Quick Start

### 1. Download & Install
```bash
git clone https://github.com/tlswa-123/sound_visualization_game.git
cd sound_visualization_game
pip install pygame numpy pyaudio scipy
```

### 2. Run the Game
```bash
python single_block_game.py
```

### 3. Test Your Microphone (Optional)
```bash
python test_audio.py
```

## 🎮 How to Play

### Game Objective
Guide the blue character (👤) from the starting position to the glowing yellow flag (🎯) by creating a path with sound-generated blocks.

### Controls & Interaction

#### 🎵 **Step 1: Make Sound to Generate Blocks**
- **Speak** into your microphone (say "hello", count numbers, etc.)
- **Whistle** or make any sound
- **Tap** on your desk near the microphone
- The first block appears automatically at the character's position

#### 🖱️ **Step 2: Click to Place More Blocks**  
- **Make sound** first (you have 5 seconds after each sound)
- **Left-click** anywhere on the game area to place a block
- Your character automatically moves to reachable blocks
- You get **15 blocks total** to reach the goal

#### ⌨️ **Step 3: Manual Movement (Optional)**
- `W` or `↑` - Move forward
- `S` or `↓` - Move backward  
- `A` or `←` - Move left
- `D` or `→` - Move right

### 🎯 **Win/Lose Conditions**
- **🎉 YOU WIN!** - Reach the yellow flag
- **💥 GAME OVER!** - Use all 15 blocks without reaching the goal
- **Press SPACE** to restart after game ends

## 🎨 Visual Features

### 🌊 **Dynamic Ocean Blocks**
- **Low frequency sounds** (<140Hz) create **blue ocean blocks**
- **Animated wave borders** that flow around the block edges
- **Sparkling water effects** with dynamic highlights

### 🏜️ **Desert Blocks** 
- **Medium frequency sounds** (140-200Hz) create **golden desert blocks**
- **Flowing sand particles** across the surface

### 🌱 **Grassland Blocks**
- **High frequency sounds** (>200Hz) create **green grass blocks**  
- **Swaying grass blades** animation on the surface

### 🎨 **Enhanced Background**
- **Multi-layer sky gradient** from light blue to pink
- **Floating clouds** that drift across the screen
- **Natural lighting effects**

## 🎵 Sound-to-Visual Mapping

| Your Sound | Block Type | Visual Effect |
|------------|------------|---------------|
| **Low voice, bass sounds** | 🌊 Ocean | Blue with wave animations |
| **Normal talking** | 🏜️ Desert | Golden with sand particles |
| **High voice, whistling** | 🌱 Grass | Green with swaying grass |
| **Volume (loudness)** | Block transparency | Louder = more solid |
| **Duration** | Block height | Longer sounds = taller blocks |

## 🎧 Audio Requirements

- **Microphone access** (the game will ask for permission)
- **Quiet environment** recommended for best results
- **Clear sounds** work better than background noise
- **No specific words needed** - any sound works!

## 🛠️ Alternative Versions

If you want to try different versions of the game:

### 🎯 Main Game (Recommended)
```bash
python single_block_game.py
```
**Best experience** - Full platform game with all features

### 🎨 Other Versions Available
```bash
python real_audio_game.py    # Continuous terrain generation
python enhanced_game.py      # Visual effects demo  
python simple_game.py        # Lightweight version
python main.py              # Full 3D version (requires more dependencies)
```

### 🪟 Windows Users
```bash
# Double-click setup.bat for guided installation
setup.bat
```

## 🎮 Game Interface

### Left Panel - Game Rules
- Shows current rules and controls
- Color coding for different block types  
- Win/lose conditions

### Center Area - Game World
- 3D isometric game environment
- Your character (blue dot with eyes)
- Goal flag (glowing yellow)
- Sound-generated blocks

### Right Panel - Real-time Info  
- **Audio levels** - Volume, frequency, duration bars
- **Game status** - Current state and progress
- **Player position** - Character coordinates

## � Background Music

The game includes **automatically generated background music**:
- **Relaxing instrumental** loops during gameplay
- **Generated on startup** if not present
- **Can be regenerated** with: `python generate_music.py`

## 🔧 Troubleshooting

### 🎤 Microphone Issues
```bash
# Test your microphone first
python test_audio.py

# Check microphone permissions in your system settings
# Windows: Settings > Privacy > Microphone
# macOS: System Preferences > Security & Privacy > Microphone
```

### 📦 Installation Problems
```bash
# If pyaudio fails to install:
pip install --upgrade pip
pip install pyaudio --force-reinstall

# Alternative for Windows:
pip install pipwin
pipwin install pyaudio
```

### 🎮 Game Not Responding to Sound
1. **Check microphone permissions**
2. **Ensure microphone is not muted**
3. **Try speaking louder or closer to microphone**
4. **Run test_audio.py to verify audio input**
5. **Close other applications using microphone**

## 📁 Project Structure

```
sound_visualization_game/
├── single_block_game.py    # 🎯 Main game (recommended)
├── generate_music.py       # 🎵 Background music generator
├── background_music.wav    # 🎼 Generated background music
├── test_audio.py          # 🎤 Microphone test tool
├── requirements.txt       # 📦 Dependencies list
├── setup.bat             # 🪟 Windows setup script
├── README.md             # 📖 This documentation
└── [other game versions] # 🎮 Alternative implementations
```

## 🚀 Technical Features

- **Real-time audio processing** with PyAudio + NumPy FFT
- **3D isometric rendering** using Pygame  
- **Dynamic block generation** based on sound characteristics
- **Smooth character movement** and pathfinding
- **Background music generation** with procedural melodies
- **Cross-platform support** (Windows/macOS/Linux)

## 🎯 Development

### Core Components
- **`RealAudioProcessor`** - Captures and analyzes microphone input
- **`TerrainBlock3D`** - Manages 3D block rendering with surface effects  
- **`Player`** - Character movement and pathfinding logic
- **`Goal`** - Target flag with glowing animations
- **`SingleBlockVisualizationGame`** - Main game loop and state management

### Contributing
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test with `python test_audio.py`
5. Submit a pull request

## 📜 License

Open source project - feel free to use, modify, and share!

## 🎉 Enjoy!

**Have fun creating sound-powered pathways and exploring the 3D world you build with your voice!** 

For issues or suggestions, please open a GitHub issue or contribute to the project.