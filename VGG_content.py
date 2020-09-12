import  numpy as np
vgg16_data = np.load('vgg16.npy', encoding='bytes')
print(type(vgg16_data))
# 是key:values的形式，所以可以转化成字典的形式
data_dict = vgg16_data.item()
print(data_dict.keys(), len(data_dict))
# conv 4个，之后接了3个全连接层

conv1_1 = data_dict[b'conv1_1']
print(len(conv1_1))  # 输出2，表示w,b
w, b = conv1_1
print(w.shape, b.shape)  # (3,3,3,64) (64,)

fc6 = data_dict[b'fc6']
print(len(fc6))
w, b = fc6
print(w.shape, b.shape)  # (25088, 4096) (4096, )