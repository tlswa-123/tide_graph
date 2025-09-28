# Kowloon Weather Bubble Animation (Real Data, Sci‑Fi Style)

A data‑driven animated visualization that turns real hourly weather data for Kowloon, Hong Kong into a glowing, sci‑fi bubble field. Great for showing data mapping, motion design, and aesthetic polish in a portfolio.

- Data source: Open‑Meteo (free weather API)
- Time window: past 5 days up to the current hour (downsampled to every 4 hours)
- Smooth transitions with interpolation between steps

## Demo

Add a short screen recording or screenshots here for your portfolio (optional):
- docs/screenshot_1.png — Main view
- docs/screenshot_2.png — Side panel & legends
- docs/animation.gif — Short loop of the animation

> Tip: On Windows you can record with Xbox Game Bar (Win+G) or use OBS; then drop the files into a `docs/` folder and update the paths above.

## Features

- Real data (Kowloon, Hong Kong) pulled from Open‑Meteo
- Clean mapping:
  - Temperature → color (light blue → light green → light yellow)
  - Humidity → bubble size
  - Wind speed → number of bubbles
  - Surface pressure → position (2D center)
- Motion design:
  - Interpolated frames between 4‑hour steps for smooth change
  - Directional spread aligned with wind direction
  - Glow layers, trails, ambient particles, subtle wind ribbons
- Side panel:
  - Clear legend and live values (time, temp, humidity, wind, pressure)
  - Temperature gradient bar
  - Sparklines (recent history)
  - Source & location labels

## Repository structure

- `spiral2.py` — main visualization (English UI; recommended to run and submit)
- `spiral.py` — earlier working version kept for reference
- `README.md` — this file
- `requirements.txt` — minimal dependencies to run the script

## Quickstart (Windows, PowerShell)

```powershell
# 1) Create and activate a virtual environment (recommended)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2) Install dependencies
python -m pip install -r requirements.txt

# 3) Run the animation
python .\spiral2.py
```

Notes:
- Uses the TkAgg backend (Matplotlib) to open a desktop window. On standard Windows Python, Tk is included by default.
- The script fetches live data; if the network call fails, it falls back to synthetic data so the animation still runs.

## How it works (concise)

- Fetch hourly weather for Kowloon via Open‑Meteo (timezone Asia/Shanghai), take the past 5 days, and filter out any future timestamps.
- Downsample to 4‑hour steps, then interpolate between consecutive steps to keep motion smooth.
- Visual encodings:
  - Temperature drives a color blend from light blue → light green → light yellow.
  - Humidity scales bubble size (with slight jitter per bubble).
  - Wind speed sets how many bubbles to draw.
  - Pressure maps to a 2D center; per‑frame positions are sampled from an anisotropic Gaussian oriented by wind direction to create a directional cluster.
- Cosmetic only (does not change data):
  - Multi‑layer glow, faint trails, ambient star‑like particles, and wind direction ribbons.
  - Rounded translucent side panel with legends/values/sparklines.

## Configuration (edit `spiral2.py`)

- Time window: search for the Open‑Meteo URL and tweak `past_days`/`forecast_days` and the filter logic.
- Step size: change `step_hours` (currently 4) to 1, 3, 6, etc.
- Smoothness vs speed:
  - `smoothing_steps` — frames between steps (higher = smoother)
  - `interval` — milliseconds per frame (lower = faster)
- Visual sensitivity:
  - Bubble size mapping in `humidity_to_size`
  - Bubble count cap in `wind_to_count`

## Troubleshooting

- A window doesn’t open / backend errors:
  - Ensure you’re running on a local desktop session (not headless).
  - Verify Matplotlib is installed correctly and Tk is available (standard Windows Python includes Tk).
- Slow performance:
  - Decrease bubble count cap (in `wind_to_count`), reduce `ambient_n`, or raise `interval` slightly.
- No network or API blocked:
  - The script will print a message and switch to synthetic data so you can still demo.

## Acknowledgements

- Weather API: [Open‑Meteo](https://open-meteo.com/)
- Matplotlib, NumPy (visualization and numerics)

## License

This coursework is submitted as part of a class assignment. If you plan to reuse or adapt this code outside of coursework, please add a proper open‑source license (e.g., MIT) or include reuse permissions here.