import matplotlib
matplotlib.use('TkAgg')

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.gridspec as gridspec
import datetime, requests

# Fonts (Windows-friendly) for Chinese labels
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# --------------------------
# Real data (Open-Meteo) - Kowloon, Hong Kong
# --------------------------
lat, lon = 22.3167, 114.1819  # Kowloon
# Use hourly data for better visible changes
url = (
    "https://api.open-meteo.com/v1/forecast"
    f"?latitude={lat}&longitude={lon}"
    "&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m,wind_direction_10m,surface_pressure"
    "&timezone=Asia%2FShanghai&past_days=1&forecast_days=2"
)

try:
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    hourly = resp.json()["hourly"]
    times = np.array(hourly["time"])  # ISO strings
    temperature_2m = np.array(hourly["temperature_2m"], dtype=float)
    humidity_2m = np.array(hourly["relative_humidity_2m"], dtype=float)
    wind_10m = np.array(hourly["wind_speed_10m"], dtype=float)
    wind_dir_10m = np.array(hourly.get("wind_direction_10m", [np.nan]*len(times)), dtype=float)
    pressure_sl = np.array(hourly["surface_pressure"], dtype=float)
except Exception as e:
    # Fallback to synthetic but plausible variation (still changes frame-to-frame)
    print("API error, fallback to synthetic data:", e)
    hours = np.arange(24 * 2)
    times = np.array([
        (datetime.datetime.now() + datetime.timedelta(hours=int(h))).strftime("%Y-%m-%dT%H:00")
        for h in hours
    ])
    rng = np.random.default_rng(42)
    temperature_2m = 26 + 3*np.sin(hours/24*2*np.pi) + rng.normal(0, 0.4, size=hours.size)
    humidity_2m = 65 + 15*np.sin((hours+6)/24*2*np.pi) + rng.normal(0, 2, size=hours.size)
    wind_10m = 12 + 6*np.sin((hours+3)/24*2*np.pi) + rng.normal(0, 1.2, size=hours.size)
    wind_dir_10m = (180 + 60*np.sin(hours/12*2*np.pi) + rng.normal(0,8,size=hours.size)) % 360
    pressure_sl = 1008 + 6*np.sin((hours+12)/48*2*np.pi) + rng.normal(0, 0.6, size=hours.size)

n_frames = len(times)
# Normalize helpers with safe ranges derived from the data window
Tmin, Tmax = float(np.nanmin(temperature_2m)), float(np.nanmax(temperature_2m))
Pmin, Pmax = float(np.nanmin(pressure_sl)), float(np.nanmax(pressure_sl))
Hmin, Hmax = 0.0, 100.0
Wmin, Wmax = max(0.0, float(np.nanmin(wind_10m))), float(np.nanmax(wind_10m))

def norm(value, vmin, vmax):
    if vmax - vmin < 1e-6:
        return 0.5
    return float((value - vmin) / (vmax - vmin))

# Color mix (light blue -> light green)
# light blue: (0.55, 0.83, 1.0); light green: (0.66, 0.94, 0.78)
C0 = np.array([0.55, 0.83, 1.00])
C1 = np.array([0.66, 0.94, 0.78])

def temp_to_color(t):
    a = norm(t, Tmin, Tmax)
    rgb = (1-a)*C0 + a*C1
    return (rgb[0], rgb[1], rgb[2], 0.70)  # add alpha

# Humidity to bubble size
def humidity_to_size(h):
    a = norm(h, Hmin, Hmax)
    return 60 + a*460  # 60..520

# Wind to bubble count
def wind_to_count(w):
    # Keep it data-driven but visible
    return int(np.clip(10 + w*2.0, 12, 140))

# Pressure to cluster center (position)
# Map pressure to polar center (r, theta) to get a 2D position, then bubbles scatter around the center

def pressure_to_center(p):
    a = norm(p, Pmin, Pmax)
    theta = 2*np.pi * a  # wrap around across range
    r = 0.2 + 1.6 * a    # 0.2..1.8
    return r*np.cos(theta), r*np.sin(theta)

# --------------------------
# Figure with side panel for legend and values
# --------------------------
fig = plt.figure(figsize=(12, 7), facecolor='black')
gs = gridspec.GridSpec(1, 2, width_ratios=[3.0, 1.15], wspace=0.06)
ax = fig.add_subplot(gs[0, 0])
ax_panel = fig.add_subplot(gs[0, 1])
for a in (ax, ax_panel):
    a.set_facecolor('black')

ax.set_xlim(-2.2, 2.2)
ax.set_ylim(-1.6, 1.6)
ax.set_xticks([])
ax.set_yticks([])
ax.set_title("Kowloon Hourly Weather — 气泡图 (真实数据)", color='#aeeaff', fontsize=13, pad=10)

# Draw small sparklines to show recent changes

def sparkline(axp, x0, y0, w, h, data, color, label):
    axp.plot([x0, x0+w], [y0, y0], color='#22424d', lw=1)
    if len(data) < 2:
        return
    d = np.array(data, dtype=float)
    vmin, vmax = np.nanmin(d), np.nanmax(d)
    if vmax - vmin < 1e-6:
        vmax = vmin + 1.0
    xs = np.linspace(x0, x0+w, len(d))
    ys = y0 + (d - vmin) / (vmax - vmin) * h
    axp.plot(xs, ys, color=color, lw=1.5)
    axp.text(x0, y0+h+0.005, f"{label}", color=color, fontsize=8)

# Pre-draw panel explanation

def draw_panel(frame_idx):
    ax_panel.clear()
    ax_panel.set_facecolor('black')
    ax_panel.set_xticks([])
    ax_panel.set_yticks([])
    ax_panel.set_xlim(0, 1)
    ax_panel.set_ylim(0, 1)

    ax_panel.text(0.04, 0.95, "可视化映射", color='#7fffd4', fontsize=12, weight='bold', va='top')
    ax_panel.text(0.06, 0.88, "温度 → 颜色 (淡蓝→淡绿)", color='#cfefff', fontsize=9)
    ax_panel.text(0.06, 0.83, "湿度 → 气泡大小", color='#cfefff', fontsize=9)
    ax_panel.text(0.06, 0.78, "风速 → 气泡数量", color='#cfefff', fontsize=9)
    ax_panel.text(0.06, 0.73, "气压 → 位置 (二维)", color='#cfefff', fontsize=9)

    # Temperature gradient bar
    gx0, gx1, gy = 0.06, 0.94, 0.67
    steps = 60
    for i in range(steps):
        a = i/(steps-1)
        col = (1-a)*C0 + a*C1
        ax_panel.plot([gx0 + (gx1-gx0)*a, gx0 + (gx1-gx0)*a], [gy, gy+0.035], color=col, solid_capstyle='butt', linewidth=3)
    ax_panel.text(gx0, gy+0.05, "低温", color='#9bd0ff', fontsize=8)
    ax_panel.text(gx1-0.06, gy+0.05, "高温", color='#bdf5d5', fontsize=8)

    # Current values
    t = float(temperature_2m[frame_idx])
    h = float(humidity_2m[frame_idx])
    w = float(wind_10m[frame_idx])
    p = float(pressure_sl[frame_idx])

    ax_panel.text(0.04, 0.60, "当前数值", color='#7fffd4', fontsize=12, weight='bold')
    ax_panel.text(0.06, 0.54, f"时间: {times[frame_idx].replace('T',' ')}", color='#cfefff', fontsize=9)
    ax_panel.text(0.06, 0.49, f"温度: {t:.1f}°C", color='#aeeaff', fontsize=10)
    ax_panel.text(0.06, 0.44, f"湿度: {h:.0f}%", color='#aeeaff', fontsize=10)
    ax_panel.text(0.06, 0.39, f"风速: {w:.1f} km/h", color='#aeeaff', fontsize=10)
    ax_panel.text(0.06, 0.34, f"气压: {p:.0f} hPa", color='#aeeaff', fontsize=10)

    # Sparklines (last 36 hours)
    N = 36
    s0 = max(0, frame_idx-N+1)
    idx = slice(s0, frame_idx+1)
    sparkline(ax_panel, 0.06, 0.26, 0.88, 0.05, temperature_2m[idx], '#bdf5d5', '温度')
    sparkline(ax_panel, 0.06, 0.19, 0.88, 0.05, humidity_2m[idx], '#9bd0ff', '湿度')
    sparkline(ax_panel, 0.06, 0.12, 0.88, 0.05, wind_10m[idx], '#7fffd4', '风速')
    sparkline(ax_panel, 0.06, 0.05, 0.88, 0.05, pressure_sl[idx], '#74c5ff', '气压')

    ax_panel.text(0.04, 0.01, "来源: Open‑Meteo 小时级 | 坐标: Kowloon", color='#6fbadf', fontsize=8)

# Keep ghost trails for sci‑fi feel (without altering data mapping)
trails = []  # each: (xs, ys, alpha)
max_trails = 8

# Animation frame

def animate(frame_idx):
    ax.clear()
    ax.set_facecolor('black')
    ax.set_xlim(-2.2, 2.2)
    ax.set_ylim(-1.6, 1.6)
    ax.set_xticks([])
    ax.set_yticks([])

    # Read current hour
    t = float(temperature_2m[frame_idx])
    h = float(humidity_2m[frame_idx])
    w = float(wind_10m[frame_idx])
    p = float(pressure_sl[frame_idx])
    wd = float(wind_dir_10m[frame_idx]) if not np.isnan(wind_dir_10m[frame_idx]) else 0.0

    # Data-driven mappings
    bubble_count = wind_to_count(w)
    cx, cy = pressure_to_center(p)

    # Reproducible random around pressure-driven center, anisotropic along wind direction
    local_rng = np.random.default_rng(frame_idx * 99991 + 7)
    spread = 0.28 + 0.35 * (1.0 - norm(p, Pmin, Pmax))  # wider when pressure is lower end
    theta = np.deg2rad(wd)
    # Major/minor scales (keep subtle)
    sigma_major = spread * (1.0 + 0.3*norm(w, Wmin, Wmax))
    sigma_minor = spread * 0.6
    u = local_rng.normal(0, sigma_major, size=bubble_count)
    v = local_rng.normal(0, sigma_minor, size=bubble_count)
    x_off = u*np.cos(theta) - v*np.sin(theta)
    y_off = u*np.sin(theta) + v*np.cos(theta)
    xs = cx + x_off
    ys = cy + y_off

    # Humidity -> size (with slight per-bubble jitter)
    base_size = humidity_to_size(h)
    jitter = local_rng.normal(1.0, 0.06, size=bubble_count)
    sizes = np.clip(base_size * jitter, 40, 560)

    # Temperature -> color
    col = temp_to_color(t)
    cols = np.tile(np.array(col), (bubble_count, 1))

    # Draw glow layers for sci‑fi
    for scale, a in [(1.5, 0.09), (1.2, 0.18), (1.0, 0.72)]:
        ax.scatter(xs, ys, s=sizes*scale, c=cols, alpha=a, edgecolors='none')

    # Update and draw faint trails
    trails.append((xs.copy(), ys.copy(), 0.16))
    if len(trails) > max_trails:
        trails.pop(0)
    for i, (tx, ty, alpha) in enumerate(trails):
        fade = alpha * (i+1)/len(trails)
        ax.scatter(tx, ty, s=16, c=[[0.0, 1.0, 1.0, fade]], edgecolors='none')

    # Title per frame
    ax.text(0.02, 1.52, f"Kowloon Hourly Weather  |  {times[frame_idx].replace('T',' ')}",
            color='#aeeaff', transform=ax.transAxes, fontsize=11)

    # Right panel
    draw_panel(frame_idx)

    return []

# Animate all hours; smooth interval
ani = FuncAnimation(fig, animate, frames=n_frames, interval=90, blit=False, repeat=True)
plt.show()



