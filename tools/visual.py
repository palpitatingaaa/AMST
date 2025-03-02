import mmcv
from mmseg.apis import inference_segmentor, init_segmentor
from mmseg.core.evaluation import get_palette
import matplotlib.pyplot as plt

# 配置文件和检查点文件
config_file = '/home/SegFormer-master/out_model/segformer.b1.1024x1024.city.160k/segformer.b1.1024x1024.city.160k.py'
checkpoint_file = '/home/SegFormer-master/out_model/segformer.b1.1024x1024.city.160k/iter_264000.pth'

# 初始化分割模型
model = init_segmentor(config_file, checkpoint_file, device='cuda:0')

# 待预测的图像
img = '/home/SegFormer-master/data/cityscapes/leftImg8bit/test/mainz/mainz_000000_011339_leftImg8bit.png'

# 推理分割
result = inference_segmentor(model, img)

# 可视化结果
# 获取模型的调色板
palette = get_palette('cityscapes')

# 保存叠加结果
model.show_result(img, result, palette=palette, show=False, out_file='/home/SegFormer-master/1937output.jpg')

# 或者直接显示分割图
plt.imshow(result[0])
plt.axis('off')
plt.show()
