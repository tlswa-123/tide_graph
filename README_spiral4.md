# Spiral4 Weather Visualization

## 📊 Overview
Advanced weather data visualization for Kowloon, Hong Kong with enhanced visual effects and real-time data integration.

## 🚀 Features
- **Real-time Data**: Fetches weather data from Open-Meteo API for Kowloon
- **Time Range**: Past 3 days with 4-hour intervals
- **Fast Animation**: Quick transitions (35ms intervals, 5 frames per step)
- **Enhanced Effects**:
  - White-to-golden cloud particles (60-140 particles based on cloud cover)
  - Expanding ripple effects around bubbles
  - Individual bubble ring animations
  - Dramatic background brightness modulation
  - Solar radiation golden glow effects

## 🎯 Data Mappings
- **Temperature** → Bubble color (blue → green → yellow)
- **Humidity** → Bubble size (larger = more humid)
- **Wind Speed** → Number of bubbles (more wind = more bubbles)
- **Air Pressure** → Bubble position (high pressure = center clustering)
- **Cloud Cover** → Background darkness + white-golden particles
- **Solar Radiation** → Golden glow intensity

## 📁 Files
- `spiral4.py` - Main visualization script
- `spiral4_weather_animation.gif` - Generated animation (6.3MB, 60 frames)

## 🛠️ Requirements
```bash
pip install matplotlib numpy requests pillow
```

## ▶️ Usage
```bash
python spiral4.py
```

The script will:
1. Fetch real weather data from Open-Meteo API
2. Display the interactive animation window
3. Automatically save a GIF animation (spiral4_weather_animation.gif)

## 🎨 Visual Enhancements
- Full-screen layout (16×10 aspect ratio)
- Separated information panel with 4 sections:
  - Mapping explanations
  - Current weather values
  - Sparkline trends
  - Data source info
- Sci-fi aesthetic with glow effects and trails
- Responsive cloud effects with particle systems

## 📊 Performance
- Optimized particle count for balance between visual impact and performance
- Fast animation cycle (~3-4 seconds per complete loop)
- Efficient data interpolation for smooth transitions

---
*Generated on September 29, 2025*