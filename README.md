# Kowloon Weather Bubble Animation (Real Data, Sci‑Fi Style)

A data‑driven animated visualization that turns real hourly weather data for Kowloon, Hong Kong into a glowing, sci‑fi bubble field with advanced atmospheric effects. Evolved through multiple iterations to achieve professional-grade visual aesthetics and real-time data integration.

- Data source: Open‑Meteo (free weather API)
- Time window: **past 3 days** (optimized from 5 days), every 4 hours
- **Ultra-fast transitions** with enhanced interpolation (35ms intervals, 5 frames per step)
- **Latest version: spiral4.py** with dramatic weather effects and GIF export

## Demo

**Latest Animation**: `spiral4_weather_animation.gif` (6.3MB, 60 frames)

Visual progression through project iterations:
- `spiral.py` — Original spiral pattern visualization
- `spiral2.py` — Enhanced bubble chart with full-screen layout
- `spiral4.py` — **Final version** with dramatic atmospheric effects

> **New Features in spiral4**: White-to-golden cloud particles, expanding bubble rings, ultra-fast animation, automatic GIF export

## Features

### Core Data Visualization
- **Real-time data** (Kowloon, Hong Kong) from Open‑Meteo API
- **Enhanced data mapping**:
  - Temperature → color gradient (blue → green → yellow)
  - Humidity → bubble size (80-980 pixel range)
  - Wind speed → bubble count (dynamic clustering)
  - Surface pressure → bubble positioning (anisotropic distribution)
  - **Cloud cover** → background brightness + white-to-golden particles
  - **Solar radiation** → golden glow intensity effects

### Advanced Visual Effects (spiral4)
- **Dramatic weather responses**:
  - White-to-golden particle systems (60-140 particles based on cloud cover)
  - Multi-layer expanding ripples around pressure centers
  - Individual bubble ring animations (3 expanding rings per bubble)
  - Background brightness modulation (0.2-1.0 range for high contrast)
- **Performance optimizations**:
  - Ultra-fast animation (35ms intervals, 2.3x speed increase)
  - Reduced time window (3 days vs 5 days for faster cycles)
  - Optimized particle counts for visual impact without lag

### Professional UI Design
- **Full-screen layout** (16×10 aspect ratio, no black borders)
- **Separated information panel** with 4 distinct sections:
  - **Mapping**: Data variable explanations with color coding
  - **Current**: Real-time weather values with large, readable text
  - **Sparklines**: Historical trend charts (expanded to 18% of panel height)
  - **Source**: API attribution and location information
- **Sci-fi aesthetic**: Glow effects, trails, ambient particles, corner brackets
- **Enhanced typography**: Larger fonts, better contrast, professional styling

## Repository Structure

### Main Files
- **`spiral4.py`** — **Latest version** with enhanced effects and GIF export ⭐
- **`spiral4_weather_animation.gif`** — Generated animation (6.3MB, 60 frames)
- **`README_spiral4.md`** — Detailed documentation for spiral4 features
- `spiral2.py` — Full-screen version with separated panels (stable backup)
- `spiral.py` — Original spiral pattern (kept for reference)

### Project Evolution
```
spiral.py (original)
    ↓ Enhanced UI + Real data integration
spiral2.py (full-screen + panels)
    ↓ Speed optimization + Dramatic effects
spiral4.py (final version)
```

### Documentation & Dependencies
- `README.md` — This comprehensive guide
- `requirements.txt` — Dependencies: matplotlib, numpy, requests, pillow

## Quickstart (Windows, PowerShell)

```powershell
# 1) Create and activate a virtual environment (recommended)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2) Install dependencies (including pillow for GIF export)
python -m pip install -r requirements.txt

# 3) Run the latest version (recommended)
python .\spiral4.py
# This will display the animation AND automatically save spiral4_weather_animation.gif

# Alternative: Run stable version
python .\spiral2.py
```

Notes:
- Uses the TkAgg backend (Matplotlib) to open a desktop window. On standard Windows Python, Tk is included by default.
- The script fetches live data; if the network call fails, it falls back to synthetic data so the animation still runs.

## How it Works (Technical Overview)

### Data Pipeline
- **API Integration**: Fetch hourly weather for Kowloon via Open‑Meteo (timezone Asia/Shanghai)
- **Time Window**: Past 3 days (optimized from 5 days for faster cycles)
- **Sampling**: Downsample to 4‑hour steps, then interpolate with 5 frames per step
- **Fallback**: Synthetic data generation if API unavailable

### Visual Encoding System
- **Temperature**: Color gradient mapping (blue → green → yellow) with smooth transitions
- **Humidity**: Bubble size scaling (80-980 pixels) with per-bubble variation
- **Wind Speed**: Dynamic bubble count with directional clustering
- **Pressure**: 2D center positioning via anisotropic Gaussian distribution
- **Cloud Cover**: Background brightness (0.2-1.0) + white-to-golden particle effects
- **Solar Radiation**: Golden glow layers with intensity variation

### Advanced Effect Systems (spiral4)
- **Particle Physics**: White-to-golden gradient particles (60-140 based on cloud intensity)
- **Ring Animations**: Individual bubble expansion rings (3 layers per bubble)
- **Atmospheric Effects**: Multi-layer ripples around pressure centers
- **Performance**: Optimized rendering with 2.3x speed increase over previous versions

## Project Evolution & Improvements

### Version History
| Version | Key Features | Performance | Visual Effects |
|---------|-------------|-------------|----------------|
| `spiral.py` | Original spiral pattern, basic data mapping | Standard | Basic glow effects |
| `spiral2.py` | Full-screen layout, separated panels, real data | 75ms intervals | Enhanced trails, sparklines |
| `spiral4.py` | **Final version** with dramatic effects | 35ms intervals (**2.3x faster**) | White-golden particles, expanding rings |

### Major Improvements Timeline
1. **Data Integration**: From synthetic → real Open-Meteo API data
2. **UI Evolution**: From basic plot → full-screen with separated information panels
3. **Visual Enhancement**: Added cloud cover & solar radiation mappings
4. **Performance Optimization**: 5 days → 3 days, 12 frames → 5 frames per step
5. **Atmospheric Effects**: White-to-golden particle systems with dramatic cloud responses
6. **Export Capability**: Automatic GIF generation with optimized settings

### Technical Achievements
- **Real-time data integration** with API fallback system
- **Responsive visual effects** that react to weather conditions
- **Professional UI design** with separated information sections
- **Performance optimization** for smooth real-time animation
- **Cross-platform compatibility** with robust dependency management

## Configuration Options

### Time & Animation Settings (spiral4.py)
- **Time window**: Modify `past_days=3` in Open‑Meteo URL for different historical ranges
- **Step size**: Change `step_hours = 4` to 1, 3, 6, etc. for different temporal granularity
- **Animation speed**:
  - `smoothing_steps = 5` — frames between steps (current: ultra-fast)
  - `interval = 35` — milliseconds per frame (current: 35ms for 2.3x speed boost)

### Visual Effects Tuning
- **Particle systems**: Adjust `cloud_particles` range (60-140) in cloudy weather effects
- **Bubble sizing**: Modify `humidity_to_size` function (current: 80-980 pixel range)
- **Bubble count**: Change cap in `wind_to_count` (affects performance vs visual density)
- **Color gradients**: Customize temperature color mapping in `temp_to_color`

### GIF Export Settings
- **Frame limit**: `gif_frames = min(n_frames, 60)` for file size control
- **Quality**: `fps=10, dpi=80` balance between quality and file size
- **Output**: Automatically saved as `spiral4_weather_animation.gif`

## Troubleshooting

- A window doesn’t open / backend errors:
  - Ensure you’re running on a local desktop session (not headless).
  - Verify Matplotlib is installed correctly and Tk is available (standard Windows Python includes Tk).
- Slow performance:
  - Decrease bubble count cap (in `wind_to_count`), reduce `ambient_n`, or raise `interval` slightly.
- No network or API blocked:
  - The script will print a message and switch to synthetic data so you can still demo.

## Acknowledgements

### Data & APIs
- **Weather API**: [Open‑Meteo](https://open-meteo.com/) - Free weather data with excellent coverage
- **Location**: Kowloon, Hong Kong (22.3167°N, 114.1819°E)

### Technologies & Libraries
- **Matplotlib**: Advanced 2D plotting and animation framework
- **NumPy**: Numerical computing and array operations
- **Requests**: HTTP library for API data fetching
- **Pillow**: Image processing for GIF export functionality

### Development Notes
- **Iterative Design**: Evolved through multiple versions based on performance and visual feedback
- **Real-time Optimization**: Balanced visual quality with animation performance
- **Cross-platform**: Developed and tested on Windows with PowerShell environment

## License

This coursework is submitted as part of a class assignment. If you plan to reuse or adapt this code outside of coursework, please add a proper open‑source license (e.g., MIT) or include reuse permissions here.