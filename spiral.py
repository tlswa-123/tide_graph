import matplotlib
matplotlib.use('TkAgg')

# matplotlib 多维天气数据动态螺旋动画
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import Normalize

# 随机生成多维天气数据（温度、湿度、风速、气压）


# 生成一周（7天）天气数据，每天一组
import datetime
num_days = 7
num_points = 400
theta = np.linspace(0, 8 * np.pi, num_points)
base_r = np.linspace(0.5, 1.2, num_points)
np.random.seed(123)
dates = [(datetime.date.today() - datetime.timedelta(days=num_days-1-i)).strftime('%Y-%m-%d') for i in range(num_days)]
temperature_week = [np.linspace(10 + i, 35 - i, num_points) + np.random.normal(0, 2, num_points) for i in range(num_days)]
humidity_week = [np.linspace(30 + i, 90 - i, num_points)[::-1] + np.random.normal(0, 2, num_points) for i in range(num_days)]
wind_week = [np.abs(np.sin(np.linspace(0, 8 * np.pi, num_points))) * (10 + i) + np.random.normal(0, 1, num_points) for i in range(num_days)]
pressure_week = [np.linspace(990 + i, 1020 - i, num_points) + np.random.normal(0, 1, num_points) for i in range(num_days)]




fig, ax = plt.subplots(figsize=(7, 7), facecolor='black')
ax.set_facecolor('black')
ax.axis('off')
# 设置显示范围，确保所有螺旋都在可见区域
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)

# 画四条螺旋线，分别用不同颜色表示不同维度
spiral_temp, = ax.plot([], [], lw=2, color='red', label='Temperature')
spiral_hum, = ax.plot([], [], lw=2, color='blue', label='Humidity')
spiral_wind, = ax.plot([], [], lw=2, color='green', label='Wind')
spiral_pres, = ax.plot([], [], lw=2, color='orange', label='Pressure')




# 动画：每帧显示一天的数据，循环播放
from matplotlib.animation import FuncAnimation
def animate(day):
    temperature = temperature_week[day % num_days]
    humidity = humidity_week[day % num_days]
    wind = wind_week[day % num_days]
    pressure = pressure_week[day % num_days]

    r_temp = base_r + 0.1 * np.sin(theta) + 0.05 * (temperature - 10) / 25
    r_hum = base_r + 0.1 * np.sin(theta) + 0.05 * (humidity - 30) / 60
    r_wind = base_r + 0.1 * np.sin(theta) + 0.05 * (wind / 15)
    r_pres = base_r + 0.1 * np.sin(theta) + 0.05 * (pressure - 990) / 30

    x_temp = r_temp * np.cos(theta)
    y_temp = r_temp * np.sin(theta)
    x_hum = r_hum * np.cos(theta)
    y_hum = r_hum * np.sin(theta)
    x_wind = r_wind * np.cos(theta)
    y_wind = r_wind * np.sin(theta)
    x_pres = r_pres * np.cos(theta)
    y_pres = r_pres * np.sin(theta)

    spiral_temp.set_data(x_temp, y_temp)
    spiral_hum.set_data(x_hum, y_hum)
    spiral_wind.set_data(x_wind, y_wind)
    spiral_pres.set_data(x_pres, y_pres)
    ax.set_title(f"Weather Spiral - {dates[day % num_days]}", color='white')
    return spiral_temp, spiral_hum, spiral_wind, spiral_pres

ani = FuncAnimation(fig, animate, frames=num_days, interval=1200, blit=True, repeat=True)
plt.legend(loc='upper right', facecolor='black', labelcolor='white')
plt.show()


