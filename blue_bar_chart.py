
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# 示例数据
labels = ['A', 'B', 'C', 'D']
values = [10, 24, 36, 18]

fig, ax = plt.subplots()
bar_container = ax.bar(labels, [0]*len(values), color='#3498db')
ax.set_title('蓝色主题柱状图动画')
ax.set_xlabel('类别')
ax.set_ylabel('数值')
ax.grid(axis='y', linestyle='--', alpha=0.7)

def animate(frame):
	for bar, h in zip(bar_container, [v*frame/30 for v in values]):
		bar.set_height(h)
	return bar_container

ani = animation.FuncAnimation(fig, animate, frames=31, interval=80, blit=True)
ani.save('blue_bar_chart.gif', writer='pillow')
