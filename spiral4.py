import matplotlib
matplotlib.use('TkAgg')

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import datetime, requests

# Fonts (Windows-friendly)
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

# --------------------------
# Real data (Open-Meteo) - Kowloon, Hong Kong
# --------------------------
lat, lon = 22.3167, 114.1819  # Kowloon
# Use hourly data; time span: past 3 days
url = (
    "https://api.open-meteo.com/v1/forecast"
    f"?latitude={lat}&longitude={lon}"
    "&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m,wind_direction_10m,surface_pressure,cloudcover,shortwave_radiation"
    "&timezone=Asia%2FShanghai&past_days=3&forecast_days=1"
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
    cloudcover = np.array(hourly.get("cloudcover", [50]*len(times)), dtype=float)  # 0-100%
    shortwave_rad = np.array(hourly.get("shortwave_radiation", [0]*len(times)), dtype=float)  # W/m²
    # Keep only past data up to current hour in Asia/Shanghai
    now_sh = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=8)
    now_sh = now_sh.replace(minute=0, second=0, microsecond=0)
    now_str = now_sh.strftime("%Y-%m-%dT%H:%M")
    mask = times <= now_str
    if mask.sum() >= 2:
        times = times[mask]
        temperature_2m = temperature_2m[mask]
        humidity_2m = humidity_2m[mask]
        wind_10m = wind_10m[mask]
        wind_dir_10m = wind_dir_10m[mask]
        pressure_sl = pressure_sl[mask]
except Exception as e:
    # Fallback to synthetic but plausible variation (still changes frame-to-frame)
    print("API error, fallback to synthetic data:", e)
    hours = np.arange(-24 * 3 + 1, 1)  # past 3 days up to now
    base = datetime.datetime.now()
    times = np.array([
        (base + datetime.timedelta(hours=int(h))).strftime("%Y-%m-%dT%H:00")
        for h in hours
    ])
    rng = np.random.default_rng(42)
    temperature_2m = 26 + 3*np.sin(hours/24*2*np.pi) + rng.normal(0, 0.4, size=hours.size)
    humidity_2m = 65 + 15*np.sin((hours+6)/24*2*np.pi) + rng.normal(0, 2, size=hours.size)
    wind_10m = 12 + 6*np.sin((hours+3)/24*2*np.pi) + rng.normal(0, 1.2, size=hours.size)
    wind_dir_10m = (180 + 60*np.sin(hours/12*2*np.pi) + rng.normal(0,8,size=hours.size)) % 360
    pressure_sl = 1008 + 6*np.sin((hours+12)/48*2*np.pi) + rng.normal(0, 0.6, size=hours.size)
    cloudcover = 40 + 35*np.sin((hours+8)/24*2*np.pi) + rng.normal(0, 8, size=hours.size)
    shortwave_rad = np.maximum(0, 400*np.sin(hours/24*np.pi)**2 + rng.normal(0, 50, size=hours.size))

# Downsample to every 4 hours
step_hours = 4
idx_ds = np.arange(0, len(times), step_hours, dtype=int)
if idx_ds.size >= 2:
    times = times[idx_ds]
    temperature_2m = temperature_2m[idx_ds]
    humidity_2m = humidity_2m[idx_ds]
    wind_10m = wind_10m[idx_ds]
    wind_dir_10m = wind_dir_10m[idx_ds]
    pressure_sl = pressure_sl[idx_ds]
    cloudcover = cloudcover[idx_ds]
    shortwave_rad = shortwave_rad[idx_ds]

n_hours = len(times)
# Smoothing: interpolate between hours - further reduced for much faster animation
smoothing_steps = 5  # frames per 4-hour step (reduced from 8 for even faster changes)
n_frames = max(1, (n_hours - 1) * smoothing_steps)

# Normalize helpers with safe ranges derived from the data window
Tmin, Tmax = float(np.nanmin(temperature_2m)), float(np.nanmax(temperature_2m))
Pmin, Pmax = float(np.nanmin(pressure_sl)), float(np.nanmax(pressure_sl))
Hmin, Hmax = 0.0, 100.0
Wmin, Wmax = max(0.0, float(np.nanmin(wind_10m))), float(np.nanmax(wind_10m))
Cmin, Cmax = 0.0, 100.0  # cloud cover 0-100%
Rmin, Rmax = max(0.0, float(np.nanmin(shortwave_rad))), max(1.0, float(np.nanmax(shortwave_rad)))

def norm(value, vmin, vmax):
    if vmax - vmin < 1e-6:
        return 0.5
    return float((value - vmin) / (vmax - vmin))

# Color mix: light blue -> light green -> light yellow (keep existing low colors unchanged, only extend to yellow at high end)
C0 = np.array([0.55, 0.83, 1.00])   # light blue
C1 = np.array([0.66, 0.94, 0.78])   # light green
C2 = np.array([1.00, 0.96, 0.70])   # light yellow

# Background gradient image (subtle vignette/teal glow) - will be modulated by cloud cover
BG_XLIM = (-3.5, 3.5)  # Much larger area
BG_YLIM = (-2.5, 2.5)
nx, ny = 320, 240
xs = np.linspace(BG_XLIM[0], BG_XLIM[1], nx)
ys = np.linspace(BG_YLIM[0], BG_YLIM[1], ny)
XX, YY = np.meshgrid(xs, ys)
rx = XX / BG_XLIM[1]
ry = YY / BG_YLIM[1]
R = np.sqrt((XX/3.5)**2 + (YY/2.5)**2)
R = np.clip(R, 0.0, 2.5)
G = np.clip(1.1 - R, 0.0, 1.0)
BG_BASE_CENTER = np.array([0.02, 0.10, 0.16])  # deep teal (base)
BG_BASE_EDGE = np.array([0.0, 0.0, 0.0])       # black (base)
BG_IMG_BASE = (G[..., None] * BG_BASE_CENTER + (1.0 - G[..., None]) * BG_BASE_EDGE)
BG_IMG_BASE = np.clip(BG_IMG_BASE, 0.0, 1.0)

# Ambient particles (twinkling) precompute - MORE particles for larger area
rng_vis = np.random.default_rng(12345)
ambient_n = 450  # Much more particles for larger area
ambient_x = rng_vis.uniform(BG_XLIM[0], BG_XLIM[1], size=ambient_n)
ambient_y = rng_vis.uniform(BG_YLIM[0], BG_YLIM[1], size=ambient_n)
ambient_size = rng_vis.uniform(4, 14, size=ambient_n)  # Slightly larger particles
ambient_phase = rng_vis.uniform(0, 2*np.pi, size=ambient_n)
ambient_speed = rng_vis.uniform(0.06, 0.14, size=ambient_n)  # modulates twinkle rate

def temp_to_color(t):
    a = norm(t, Tmin, Tmax)
    if a <= 0.5:
        a2 = a * 2.0
        rgb = (1-a2)*C0 + a2*C1
    else:
        a2 = (a - 0.5) * 2.0
        rgb = (1-a2)*C1 + a2*C2
    return (rgb[0], rgb[1], rgb[2], 0.70)  # add alpha

# Humidity to bubble size (slightly larger range to emphasize differences)

def humidity_to_size(h):
    a = norm(h, Hmin, Hmax)
    return 80 + a*900  # 80..980 larger base for nicer presence

# Wind to bubble count

def wind_to_count(w):
    # Data-driven but visible; keep within reasonable bounds
    return int(np.clip(10 + w*2.0, 12, 160))

# Pressure to cluster center (position)

def pressure_to_center(p):
    a = norm(p, Pmin, Pmax)
    theta = 2*np.pi * a  # wrap around across range
    r = 0.2 + 1.6 * a    # 0.2..1.8
    return r*np.cos(theta), r*np.sin(theta)

# --------------------------
# Figure with side panel for legend and values
# --------------------------
fig = plt.figure(figsize=(16, 10), facecolor='black')

gs = gridspec.GridSpec(1, 2, width_ratios=[2.8, 1.2], wspace=0.02)
ax = fig.add_subplot(gs[0, 0])
ax_panel = fig.add_subplot(gs[0, 1])
for a in (ax, ax_panel):
    a.set_facecolor('black')

ax.set_xlim(BG_XLIM)
ax.set_ylim(BG_YLIM)
ax.set_xticks([])
ax.set_yticks([])
ax.set_title("Kowloon Weather (Past 3 Days, 4-hour steps) - Bubble Chart (Real Data)", color='#aeeaff', fontsize=16, pad=15)
# Remove axes spines and make plot fill entire area
for spine in ax.spines.values():
    spine.set_visible(False)
ax.set_position([0, 0, 0.74, 1.0])  # left, bottom, width, height - fill left 74% of figure

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

def draw_panel(hour_idx):
    ax_panel.clear()
    ax_panel.set_facecolor('black')
    ax_panel.set_xticks([])
    ax_panel.set_yticks([])
    ax_panel.set_xlim(0, 1)
    ax_panel.set_ylim(0, 1)
    # Remove axes spines and expand panel to fill right area
    for spine in ax_panel.spines.values():
        spine.set_visible(False)
    ax_panel.set_position([0.76, 0, 0.24, 1.0])  # Fill right 24% of figure completely

    # Much larger panel background with rounded corners
    panel_patch = FancyBboxPatch((0.01, 0.01), 0.98, 0.98,
                                 boxstyle="round,pad=0.015,rounding_size=0.015",
                                 facecolor=(0.00, 0.06, 0.10, 0.38),
                                 edgecolor=(0.30, 0.85, 1.00, 0.32), linewidth=1.2)
    ax_panel.add_patch(panel_patch)

    # Much more generous margins within the larger panel box
    ml, mr, mb, mt = 0.04, 0.96, 0.04, 0.96

    # MAPPING SECTION - Top area (0.96 to 0.65)
    ax_panel.text(ml, 0.94, "Mapping", color='#7fffd4', fontsize=14, weight='bold', va='top')
    ax_panel.text(ml+0.02, 0.88, "Temperature -> Color", color='#cfefff', fontsize=11, wrap=True)
    ax_panel.text(ml+0.02, 0.84, "(light blue -> light green -> light yellow)", color='#cfefff', fontsize=10, wrap=True)
    ax_panel.text(ml+0.02, 0.79, "Humidity -> Bubble size", color='#cfefff', fontsize=11)
    ax_panel.text(ml+0.02, 0.75, "Wind speed -> Bubble count", color='#cfefff', fontsize=11)
    ax_panel.text(ml+0.02, 0.71, "Pressure -> Position (2D)", color='#cfefff', fontsize=11)
    ax_panel.text(ml+0.02, 0.67, "Cloud cover -> Background brightness", color='#cfefff', fontsize=11)
    ax_panel.text(ml+0.02, 0.63, "Solar radiation -> Golden glow", color='#cfefff', fontsize=11)

    # Temperature gradient bar - part of MAPPING section
    gx0, gx1, gy = ml+0.02, mr-0.02, 0.57
    steps = 80
    mid = int(steps/2)
    for i in range(mid):
        a = i/(mid-1)
        col = (1-a)*C0 + a*C1
        x = gx0 + (gx1-gx0) * (i/steps)
        ax_panel.plot([x, x], [gy, gy+0.035], color=col, solid_capstyle='butt', linewidth=3)
    for i in range(mid, steps):
        a = (i-mid)/(steps-mid-1 if steps-mid-1>0 else 1)
        col = (1-a)*C1 + a*C2
        x = gx0 + (gx1-gx0) * (i/steps)
        ax_panel.plot([x, x], [gy, gy+0.035], color=col, solid_capstyle='butt', linewidth=3)
    ax_panel.text(gx0, gy+0.045, "Low", color='#9bd0ff', fontsize=8, ha='left')
    ax_panel.text(gx1, gy+0.045, "High", color='#fff2a8', fontsize=8, ha='right')

    # Current values (use discrete hour index for readability)
    t = float(temperature_2m[hour_idx])
    h = float(humidity_2m[hour_idx])
    w = float(wind_10m[hour_idx])
    p = float(pressure_sl[hour_idx])
    cc_val = float(cloudcover[hour_idx])
    sr_val = float(shortwave_rad[hour_idx])

    # CURRENT SECTION - Middle-upper area (0.52 to 0.32) - COMPRESSED
    ax_panel.text(ml, 0.50, "Current", color='#7fffd4', fontsize=14, weight='bold')
    ax_panel.text(ml+0.02, 0.47, f"Time: {times[hour_idx].replace('T',' ')}", color='#cfefff', fontsize=11, wrap=True)
    ax_panel.text(ml+0.02, 0.44, f"Temperature: {t:.1f}°C", color='#aeeaff', fontsize=12)
    ax_panel.text(ml+0.02, 0.41, f"Humidity: {h:.0f}%", color='#aeeaff', fontsize=12)
    ax_panel.text(ml+0.02, 0.38, f"Wind speed: {w:.1f} km/h", color='#aeeaff', fontsize=12)
    ax_panel.text(ml+0.02, 0.35, f"Pressure: {p:.0f} hPa", color='#aeeaff', fontsize=12)
    ax_panel.text(ml+0.02, 0.32, f"Cloud cover: {cc_val:.0f}%", color='#aeeaff', fontsize=12)
    ax_panel.text(ml+0.02, 0.29, f"Solar rad: {sr_val:.0f} W/m²", color='#aeeaff', fontsize=12)

    # SPARKLINES SECTION - Expanded area (0.26 to 0.08) - MUCH BIGGER
    N = 72
    s0 = max(0, hour_idx-N+1)
    idx = slice(s0, hour_idx+1)
    sp_h = 0.025  # Taller sparklines
    sp_w = (mr-ml) - 0.04
    sp_x = ml+0.02
    sparkline(ax_panel, sp_x, 0.235, sp_w, sp_h, temperature_2m[idx], '#bdf5d5', 'Temp')
    sparkline(ax_panel, sp_x, 0.205, sp_w, sp_h, humidity_2m[idx], '#9bd0ff', 'Humidity')
    sparkline(ax_panel, sp_x, 0.175, sp_w, sp_h, wind_10m[idx], '#7fffd4', 'Wind')
    sparkline(ax_panel, sp_x, 0.145, sp_w, sp_h, pressure_sl[idx], '#74c5ff', 'Pressure')
    sparkline(ax_panel, sp_x, 0.115, sp_w, sp_h, cloudcover[idx], '#ddd8aa', 'Clouds')
    sparkline(ax_panel, sp_x, 0.085, sp_w, sp_h, shortwave_rad[idx], '#ffcc66', 'Solar')

    # SOURCE SECTION - Bottom area (0.06 to 0.01)
    ax_panel.text(ml, 0.055, "Location: Kowloon, Hong Kong", color='#6fbadf', fontsize=10)
    ax_panel.text(ml, 0.025, "Source: Open-Meteo (hourly)", color='#6fbadf', fontsize=10)

# Keep ghost trails for sci‑fi feel (without altering data mapping)
trails = []  # each: (xs, ys, alpha)
max_trails = 8

# Animation frame

def animate(frame):
    # Map smoothing frame -> hour indices and interpolation factor
    i0 = int(frame // smoothing_steps)
    i1 = min(i0 + 1, n_hours - 1)
    alpha = (frame % smoothing_steps) / smoothing_steps

    ax.clear()
    ax.set_facecolor('black')
    ax.set_xlim(BG_XLIM)
    ax.set_ylim(BG_YLIM)
    ax.set_xticks([])
    ax.set_yticks([])
    # Remove axes spines each frame
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_position([0, 0, 0.74, 1.0])  # Fill left area completely

    # Interpolate current hour values for smooth change
    def lerp(a, b, t):
        return (1-t)*a + t*b

    t = float(lerp(temperature_2m[i0], temperature_2m[i1], alpha))
    h = float(lerp(humidity_2m[i0], humidity_2m[i1], alpha))
    w = float(lerp(wind_10m[i0], wind_10m[i1], alpha))
    p = float(lerp(pressure_sl[i0], pressure_sl[i1], alpha))
    cc = float(lerp(cloudcover[i0], cloudcover[i1], alpha))  # cloud cover 0-100
    sr = float(lerp(shortwave_rad[i0], shortwave_rad[i1], alpha))  # shortwave radiation
    
    # Wind direction interpolation with angle wrap handling
    wd0 = float(wind_dir_10m[i0]) if not np.isnan(wind_dir_10m[i0]) else 0.0
    wd1 = float(wind_dir_10m[i1]) if not np.isnan(wind_dir_10m[i1]) else wd0
    d = ((wd1 - wd0 + 540) % 360) - 180  # shortest path
    wd = wd0 + alpha * d

    # Cloud cover -> background brightness modulation (MUCH MORE DRAMATIC)
    cloud_factor = norm(cc, Cmin, Cmax)  # 0=clear, 1=overcast
    bg_brightness = 0.2 + 0.8 * (1.0 - cloud_factor)  # much more dramatic: 0.2-1.0
    
    # Much stronger cloud tint for dramatic effect
    cloud_tint_intensity = 0.2 + 0.8 * cloud_factor  # 0.2-1.0 intensity
    cloud_tint = np.array([0.25, 0.25, 0.35]) * cloud_tint_intensity  # stronger gray-blue tint
    bg_img = BG_IMG_BASE * bg_brightness + cloud_tint
    bg_img = np.clip(bg_img, 0.0, 1.0)
    ax.imshow(bg_img, extent=[BG_XLIM[0], BG_XLIM[1], BG_YLIM[0], BG_YLIM[1]], origin='lower', zorder=0)

    # Cloud cover -> ambient particle density & visibility (ALWAYS VISIBLE, varying intensity)
    star_visibility = 0.3 + 0.7 * (1.0 - cloud_factor)  # always 0.3-1.0 visibility
    active_stars = int(ambient_n * star_visibility)  # always show 30-100% of stars
    if active_stars > 0:
        amb_alpha_base = (0.08 + 0.15 * (0.5 * (1.0 + np.sin(ambient_phase[:active_stars] + frame * ambient_speed[:active_stars]))))
        amb_alpha_modulated = amb_alpha_base * (0.4 + 0.6 * star_visibility)  # always visible, 0.4-1.0 alpha range
        amb_cols = np.stack([
            np.full_like(amb_alpha_modulated, 0.35), 
            np.full_like(amb_alpha_modulated, 0.95), 
            np.full_like(amb_alpha_modulated, 1.0), 
            amb_alpha_modulated
        ], axis=1)
        ax.scatter(ambient_x[:active_stars], ambient_y[:active_stars], s=ambient_size[:active_stars], c=amb_cols, edgecolors='none', zorder=0.5)



    # Data-driven mappings
    bubble_count = wind_to_count(w)
    cx, cy = pressure_to_center(p)

    # Subtle wind ribbon behind bubbles
    theta = np.deg2rad(wd)
    L = 0.7 + 0.015 * np.clip(w, Wmin, Wmax)
    for o in (-0.06, 0.0, 0.06):
        ox = o * (-np.sin(theta))
        oy = o * ( np.cos(theta))
        x0 = cx - 0.30*L*np.cos(theta) + ox
        y0 = cy - 0.30*L*np.sin(theta) + oy
        x1 = cx + 0.70*L*np.cos(theta) + ox
        y1 = cy + 0.70*L*np.sin(theta) + oy
        ax.plot([x0, x1], [y0, y1], color=(0.3, 0.95, 1.0, 0.16), linewidth=2.0, zorder=0.8)

    # Reproducible random around center; stable within each hour with slight drift across alpha
    base_seed = i0 * 99991 + 7
    local_rng = np.random.default_rng(base_seed)
    spread = 0.28 + 0.35 * (1.0 - norm(p, Pmin, Pmax))
    sigma_major = spread * (1.0 + 0.3*norm(w, Wmin, Wmax))
    sigma_minor = spread * 0.6
    u = local_rng.normal(0, sigma_major, size=bubble_count)
    v = local_rng.normal(0, sigma_minor, size=bubble_count)
    x_off = u*np.cos(theta) - v*np.sin(theta)
    y_off = u*np.sin(theta) + v*np.cos(theta)

    # Add a tiny second-seed drift for extra smoothness
    drift_rng = np.random.default_rng(base_seed + 1)
    du = drift_rng.normal(0, sigma_major*0.2, size=bubble_count)
    dv = drift_rng.normal(0, sigma_minor*0.2, size=bubble_count)
    x_off += alpha * (du*np.cos(theta) - dv*np.sin(theta))
    y_off += alpha * (du*np.sin(theta) + dv*np.cos(theta))

    xs = cx + x_off
    ys = cy + y_off

    # Humidity -> size (with slight per-bubble jitter)
    base_size = humidity_to_size(h)
    jitter = local_rng.normal(1.0, 0.06, size=bubble_count)
    sizes = np.clip(base_size * jitter, 60, 1200)

    # Temperature -> color
    col = temp_to_color(t)
    cols = np.tile(np.array(col), (bubble_count, 1))

    # Draw glow layers for sci‑fi
    for scale, a_glow in [(1.9, 0.06), (1.35, 0.14), (1.0, 0.80)]:
        ax.scatter(xs, ys, s=sizes*scale, c=cols, alpha=a_glow, edgecolors='none', zorder=1.0+scale*0.01)

    # Solar radiation -> secondary golden glow layer (ALWAYS VISIBLE, varying intensity)
    rad_intensity = norm(sr, Rmin, Rmax)  # 0=no sun, 1=strong sun
    # Always show golden glow, just varying intensity
    golden_alpha = 0.08 + 0.25 * rad_intensity  # always visible: 0.08-0.33 alpha range
    golden_color = np.array([1.0, 0.75, 0.2, golden_alpha])  # warm golden
    golden_cols = np.tile(golden_color, (bubble_count, 1))
    # Golden halo always present, size varies with radiation
    glow_size = 1.8 + 0.6 * rad_intensity  # size ranges 1.8-2.4x
    ax.scatter(xs, ys, s=sizes*glow_size, c=golden_cols, edgecolors='none', zorder=0.95)
    
    # Always add some golden particles in background, count varies with radiation
    golden_particles = int(30 + 90 * rad_intensity)  # always 30-120 particles
    gp_rng = np.random.default_rng(base_seed + 99)
    gp_x = gp_rng.uniform(BG_XLIM[0], BG_XLIM[1], golden_particles)
    gp_y = gp_rng.uniform(BG_XLIM[0], BG_XLIM[1], golden_particles)
    gp_base_size = 8 + 25 * rad_intensity  # size ranges 8-33
    gp_size = gp_rng.uniform(gp_base_size, gp_base_size + 15, golden_particles)
    gp_alpha = 0.05 + 0.2 * rad_intensity  # always visible: 0.05-0.25 alpha
    ax.scatter(gp_x, gp_y, s=gp_size, c=[[1.0, 0.8, 0.3, gp_alpha]], edgecolors='none', zorder=0.3)
    
    # MUCH MORE DRAMATIC CLOUDY WEATHER EFFECTS
    if cloud_factor > 0.2:  # Trigger at lower threshold for more visibility
        # White-to-golden particles scattered around (reduced quantity for cleaner look)
        cloud_particles = int(60 + 100 * (cloud_factor - 0.2))  # 60-140 particles when cloudy (reduced from 120-280)
        cp_rng = np.random.default_rng(base_seed + 199)
        cp_x = cp_rng.uniform(BG_XLIM[0], BG_XLIM[1], cloud_particles)
        cp_y = cp_rng.uniform(BG_YLIM[0], BG_YLIM[1], cloud_particles)
        cp_size = cp_rng.uniform(15, 50, cloud_particles)  # Larger particles
        cp_alpha = 0.20 + 0.30 * (cloud_factor - 0.2)  # 0.20-0.50 alpha - much more visible
        
        # Create white-to-golden gradient based on cloud intensity
        golden_intensity = (cloud_factor - 0.2) / 0.8  # 0.0-1.0 range
        # Interpolate from white (1,1,1) to golden (1,0.8,0.3)
        cp_r = 1.0
        cp_g = 1.0 - 0.2 * golden_intensity  # 1.0 -> 0.8
        cp_b = 1.0 - 0.7 * golden_intensity  # 1.0 -> 0.3
        ax.scatter(cp_x, cp_y, s=cp_size, c=[[cp_r, cp_g, cp_b, cp_alpha]], edgecolors='none', zorder=0.4)
        
        # Much more dramatic expanding ripple layers with golden-white colors
        ripple_intensity = (cloud_factor - 0.2) * 1.25  # 0.0-1.0 intensity
        for ripple_idx, (radius_mult, alpha_mult) in enumerate([(1.2, 1.0), (1.8, 0.8), (2.4, 0.6), (3.2, 0.45), (4.0, 0.3)]):
            ripple_alpha = 0.12 + 0.20 * ripple_intensity * alpha_mult  # Much more visible
            # Golden-white ripples that match the particles
            ripple_r = 1.0
            ripple_g = 1.0 - 0.15 * golden_intensity  # 1.0 -> 0.85
            ripple_b = 1.0 - 0.5 * golden_intensity   # 1.0 -> 0.5
            ripple_color = [ripple_r, ripple_g, ripple_b, ripple_alpha]
            # Create expanding circles around pressure center
            circle_size = 120 + radius_mult * 80 * (1.0 + 0.4 * np.sin(frame * 0.15 + ripple_idx))
            ax.scatter([cx], [cy], s=circle_size, facecolors='none', 
                      edgecolors=[ripple_color], linewidths=2.5 + ripple_intensity, zorder=0.6)

    # Subtle rim stroke (lighter than fill)
    base_rgb = np.array(col[:3])
    rim_rgb = 0.55*np.array([1.0, 1.0, 1.0]) + 0.45*base_rgb
    rim_cols = np.tile(np.array([rim_rgb[0], rim_rgb[1], rim_rgb[2], 0.55]), (bubble_count, 1))
    lw = np.clip(sizes * 0.004, 0.4, 2.4)
    ax.scatter(xs, ys, s=sizes*1.02, facecolors='none', edgecolors=rim_cols, linewidths=lw, zorder=1.22)

    # EXPANDING RING EFFECTS around each individual bubble
    for bubble_idx in range(bubble_count):
        bubble_x, bubble_y = xs[bubble_idx], ys[bubble_idx]
        bubble_size = sizes[bubble_idx]
        bubble_color = col  # Use the same color as the main bubble
        
        # Create 3 expanding rings around each bubble
        for ring_idx in range(3):
            # Each ring has different timing and size
            ring_phase = frame * 0.12 + bubble_idx * 0.3 + ring_idx * 2.0
            ring_expansion = 1.2 + ring_idx * 0.6 + 0.4 * np.sin(ring_phase)
            ring_size = bubble_size * ring_expansion
            
            # Fade rings based on expansion
            base_alpha = 0.15 - ring_idx * 0.04  # 0.15, 0.11, 0.07
            expansion_fade = max(0.0, 1.0 - (ring_expansion - 1.2) / 1.5)  # Fade as they expand
            ring_alpha = base_alpha * expansion_fade
            
            if ring_alpha > 0.01:  # Only draw if visible
                ring_color = [bubble_color[0], bubble_color[1], bubble_color[2], ring_alpha]
                ax.scatter([bubble_x], [bubble_y], s=ring_size, facecolors='none', 
                          edgecolors=[ring_color], linewidths=1.0 + ring_idx * 0.3, zorder=1.15)

    # Specular highlight (small white dot offset along wind)
    glint_size = np.clip(4 + (sizes/1200)*18, 6, 22)
    glint_theta = theta - np.deg2rad(30)
    glint_x = xs + 0.05*np.cos(glint_theta)
    glint_y = ys + 0.05*np.sin(glint_theta)
    ax.scatter(glint_x, glint_y, s=glint_size, c=[[1.0, 1.0, 1.0, 0.50]], edgecolors='none', zorder=1.28)

    # Update and draw faint trails
    trails.append((xs.copy(), ys.copy(), 0.16))
    if len(trails) > max_trails:
        trails.pop(0)
    for i, (tx, ty, alpha_t) in enumerate(trails):
        fade = alpha_t * (i+1)/len(trails)
        ax.scatter(tx, ty, s=16, c=[[0.0, 1.0, 1.0, fade]], edgecolors='none', zorder=0.9)

    # Title per frame (show discrete hour start time for stability) - LARGER
    ax.text(0.02, 0.96, f"Kowloon Weather (Past 3 Days)  |  {times[i0].replace('T',' ')}",
            color='#aeeaff', transform=ax.transAxes, fontsize=14, weight='bold')

    # Corner brackets/frame
    bracket_c = (0.20, 0.95, 1.00, 0.35)
    L = 0.06
    # bottom-left
    ax.plot([0.01, 0.01+L], [0.01, 0.01], color=bracket_c, transform=ax.transAxes, clip_on=False)
    ax.plot([0.01, 0.01], [0.01, 0.01+L], color=bracket_c, transform=ax.transAxes, clip_on=False)
    # bottom-right
    ax.plot([0.99-L, 0.99], [0.01, 0.01], color=bracket_c, transform=ax.transAxes, clip_on=False)
    ax.plot([0.99, 0.99], [0.01, 0.01+L], color=bracket_c, transform=ax.transAxes, clip_on=False)
    # top-left
    ax.plot([0.01, 0.01+L], [0.99, 0.99], color=bracket_c, transform=ax.transAxes, clip_on=False)
    ax.plot([0.01, 0.01], [0.99-L, 0.99], color=bracket_c, transform=ax.transAxes, clip_on=False)
    # top-right
    ax.plot([0.99-L, 0.99], [0.99, 0.99], color=bracket_c, transform=ax.transAxes, clip_on=False)
    ax.plot([0.99, 0.99], [0.99-L, 0.99], color=bracket_c, transform=ax.transAxes, clip_on=False)

    # Right panel (discrete hour index)
    draw_panel(i0)

    return []

# Animate all hours; much faster interval for quicker changes
ani = FuncAnimation(fig, animate, frames=n_frames, interval=35, blit=False, repeat=True)

# Save as GIF animation (limit frames for reasonable file size)
print(f"Saving GIF animation with {min(n_frames, 60)} frames...")
gif_frames = min(n_frames, 60)  # Limit to 60 frames for reasonable file size
ani_gif = FuncAnimation(fig, animate, frames=gif_frames, interval=100, blit=False, repeat=True)
ani_gif.save('spiral4_weather_animation.gif', writer='pillow', fps=10, dpi=80)
print("GIF saved as: spiral4_weather_animation.gif")

plt.show()
