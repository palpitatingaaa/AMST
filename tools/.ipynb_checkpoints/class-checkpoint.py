import cv2
import matplotlib.pyplot as plt

# 加载图像
image_path = '/1937output.jpg'
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 定义点击事件
def onclick(event):
    x, y = int(event.xdata), int(event.ydata)
    pixel_value = image_rgb[y, x]  # 获取 RGB 值
    print(f"Pixel value at ({x}, {y}): {pixel_value}")

# 显示图像
fig, ax = plt.subplots()
ax.imshow(image_rgb)
cid = fig.canvas.mpl_connect('button_press_event', onclick)
plt.show()
