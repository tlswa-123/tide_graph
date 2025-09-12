import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import random

# 波浪参数
num_layers = 4
num_points = 120
frames = 80
x = np.linspace(0, 2*np.pi, num_points)
colors = ['#003366', '#006699', '#0099CC', '#33CCFF'] # 不同深浅的蓝色

# 星光参数
num_stars = 50
star_x = np.random.uniform(0, 2*np.pi, num_stars)
star_y0 = np.random.uniform(1.2, 2.2, num_stars)
star_speed = np.random.uniform(0.01, 0.03, num_stars)
star_colors = np.random.rand(num_stars, 3)  # 随机颜色
star_sizes = np.random.uniform(15, 55, num_stars) # 随机大小
star_paths = []
for _ in range(num_stars):
    num_verts = np.random.randint(6, 10)
    angles = np.linspace(0, 2*np.pi, num_verts)
    radii = np.random.uniform(0.7, 1.3, num_verts)
    verts = np.array([radii * np.cos(angles), radii * np.sin(angles)]).T
    star_paths.append(verts)

fig, ax = plt.subplots(figsize=(8,5))
ax.set_xlim(0, 2*np.pi)
ax.set_ylim(-2, 2.5)
ax.axis('off')

lines = [ax.plot([], [], lw=3)[0] for _ in range(num_layers)]
star_scat = ax.scatter([], [], s=18, c='white', marker='*', alpha=0.8)

def animate(frame):
	ax.clear()
	ax.set_xlim(0, 2*np.pi)
	ax.set_ylim(-2, 2.5)
	ax.axis('off')
	# 绘制波浪层叠
	for i in range(num_layers):
		phase = frame*0.12 + i*np.pi/num_layers
		# 调整波浪参数
		if i < 2:
			# 降低前两层（蓝色）的高度和频率
			amp = 0.4 + 0.15*i 
			freq = 0.8 + 0.2*i
		else:
			# 后两层与前两层相似
			amp = 0.4 + 0.15*(i-2) + 0.3
			freq = 0.8 + 0.2*(i-2) + 0.4
		y = amp * np.sin(freq*x + phase) - i*0.5
		ax.fill_between(x, y, -2.5, color=colors[i], alpha=0.6)
	# 星光点点
	star_y = star_y0 - frame*star_speed
	# 只显示在画布范围内的星星
	mask = star_y > -2
	# 使用不规则圆形路径
	for i in np.where(mask)[0]:
		ax.scatter(star_x[i], star_y[i], s=star_sizes[i], c=[star_colors[i]], marker=star_paths[i], alpha=0.8, edgecolors='none')
	return []

ani = animation.FuncAnimation(fig, animate, frames=frames, interval=60, blit=False)
ani.save('wave_stars.gif', writer='pillow')
