'''不是TensorFlow下那种保存模型，恢复数据的restore方法'''
import os
import math
import numpy as np
import tensorflow as tf
from PIL import Image
import time
'''VGG图像预处理，归一化。保持预训练模型和输入一致，三个通道均值'''
VGG_MEAN = [103.939, 116.779, 123.68]
'''搭建一个VGGnet16,让后将参数从模型文件中导入进来'''
class VGGNet:
    def __init__(self, data_dict):
        self.data_dict = data_dict
    # 模式识预训练好的，在做风格变换的时候是不会去改变的，定义为常量
    def get_conv_filter(self, name):
        return tf.constant(self.data_dict[name][0], name='conv')  # 每一层序列是[w, b]

    def get_fc_weight(self, name):
        return tf.constant(self.data_dict[name][0], name='fc')

    def get_bias(self, name):
        return tf.constant(self.data_dict[name][1], name='bias')
    # 创建卷积层，池化层,全连接层(需要展成一列或者一行
    # 不是用tf.layers.conv2d()，它会创建对应的W,b,
    # 这里是用更低层接口的方法，是从外面获的W, b
    def conv_layer(self, x, name):
        with tf.name_scope(name):
            conv_w = self.get_conv_filter(name)
            conv_b = self.get_bias(name)
            h = tf.nn.conv2d(x, conv_w, [1, 1, 1, 1], padding='SAME')
            h = tf.nn.bias_add(h, conv_b)  # 权值阈相加
            b = tf.nn.relu(h)  # 激活
            return h

    def pooling_layer(self, x, name):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1],
                              padding='SAME',
                              name=name)

    def fc_layer(self, x, name, activation=tf.nn.relu):
        with tf.name_scope(name):
            fc_w = self.get_fc_weight(name)
            fc_b = self.get_bias(name)
            h = tf.matmul(x, fc_w)
            h = tf.nn.bias_add(h, fc_b)
            if activation is None:
                return h
            else:
                return activation(h)

    def faltten_layer(self, x, name):
        with tf.name_scope(name):
            # [batch_size, image_width, image_height, chaanel]
            x_shape = x.get_shape().as_list()
            dim = 1
            for d in x_shape[1:]:
                dim *= d
            x = tf.reshape(x, [-1, dim])  # 将x展开成后三维相乘的长度
            return x
    # 定义个统一的调用整个VGGnet
    def build(self, x_rgb):
        '''
        Bulid VGG-16 network structure. 13个conv加上3个all-connect
        :param x_rgb: [1, 224, 224, 3] 图片为rgb格式，3通道
        :return:
        '''
        start_time = time.time()  # 生成VGG16计算图所需时间
        print('模型构建开始。。。。')
        # 对输入图片的初始化操作，一开始的时候减去它的均值，通道拆开单独运算
        # VGGnet通道排列顺序是bgr
        r, g, b = tf.split(x_rgb, [1, 1, 1], axis=3)  # 切第3通道为3份
        x_bgr = tf.concat([b - VGG_MEAN[0],
                           g - VGG_MEAN[1],
                           r - VGG_MEAN[2]], axis=3)
        assert x_bgr.get_shape().as_list()[1:] == [224, 224, 3]

        # 将图片输入到VGG网络 取名字要和预训练模型的名字一致
        # 定义为类的变量，方便在后面风格变换的时候算内容损失的时候用
        self.conv1_1 = self.conv_layer(x_bgr, b'conv1_1')
        self.conv1_2 = self.conv_layer(self.conv1_1, b'conv1_2')
        self.pool1 = self.pooling_layer(self.conv1_2, b'pool1')

        self.conv2_1 = self.conv_layer(self.pool1, b'conv2_1')
        self.conv2_2 = self.conv_layer(self.conv2_1, b'conv2_2')
        self.pool2 = self.pooling_layer(self.conv2_2, b'pool2')

        self.conv3_1 = self.conv_layer(self.pool2, b'conv3_1')
        self.conv3_2 = self.conv_layer(self.conv3_1, b'conv3_2')
        self.conv3_3 = self.conv_layer(self.conv3_2, b'conv3_3')
        self.pool3 = self.pooling_layer(self.conv3_3, b'pool3')

        self.conv4_1 = self.conv_layer(self.pool3, b'conv4_1')
        self.conv4_2 = self.conv_layer(self.conv4_1, b'conv4_2')
        self.conv4_3 = self.conv_layer(self.conv4_2, b'conv4_3')
        self.pool4 = self.pooling_layer(self.conv4_3, b'pool4')

        self.conv5_1 = self.conv_layer(self.pool4, b'conv5_1')
        self.conv5_2 = self.conv_layer(self.conv5_1, b'conv5_2')
        self.conv5_3 = self.conv_layer(self.conv5_2, b'conv5_3')
        self.pool5 = self.pooling_layer(self.conv5_3, b'pool5')

        # 在风格转换的时候不会用到全连接层的参数，可以注释掉
        '''
        self.flatten5 = self.faltten_layer(self.pool5, b'flatten')
        self.fc6 = self.fc_layer(self.flatten5, b'fc6')
        self.fc7 = self.fc_layer(self.fc6, b'fc7')
        # 最后输出是经过softmax的概率，所以不需要对它经过relu
        self.fc8 = self.fc_layer( self.fc7, b'fc8', activation=None)
        self.prob = tf.nn.softmax(self.fc8, name='prob')
        '''
        print('构建计算图完成，所需时间：%4ds' % (time.time() - start_time))

'''
vgg16_npy_path = './vgg16.npy'
# 测试结构而已，并没有输入值
data_dict = np.load(vgg16_npy_path, encoding='bytes').item()  # 变成字典
vgg16_for_result = VGGNet(data_dict)
content = tf.placeholder(tf.float32, shape=[1, 224, 224, 3])  # 看一下搭建过程是否正确
vgg16_for_result.build(content)
'''

# 随机化初始图片为初始结果，在这张图片上进行梯度下降，看效果，均值225/2，方差20
def initial_result(shape, mean, stddev):
    initial = tf.truncated_normal(shape, mean=mean, stddev=stddev)
    return tf.Variable(initial)

# 将内容图像，风格图像读入
def read_img(img_name):
    img = Image.open(img_name)
    np_img = np.array(img)  # (224, 224, 3)
    np_img = np.array([np_img], dtype=np.int32)  # 1,224,224
    return np_img

# 计算gram矩阵
def gram_matrix(x):
    '''x是VGGNet输出的特征，shape:[1,width,height,channnel]'''
    b, w, h, ch = x.get_shape().as_list()
    features = tf.reshape(x, [b, h*w, ch])
    # [h*w,ch] matrix -> [ch, h*w] * [h*w, ch] -> [ch, ch],，两两相似度
    gram = tf.matmul(features, features, adjoint_a=True) / tf.constant(ch * w * h, tf.float32)
    return gram


# 输入，输出，中间超参数
vgg16_npy_path = './vgg16.npy'
content_img_path = './source_images/xizangscorp.jpg'
style_img_path = './source_images/xikongscorp.png'
num_steps = 100
learning_rate = 10
# 内容损失和风格损失加权
lambda_c = 0.1
lambda_s = 500
# 对图片随机梯度下降，在每一步都可以输出来看看
output_dir = './run_style_transfer'
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

# 调用VGGnet进行风格处理
result = initial_result((1, 224, 224, 3), 127.5, 20)
content_val = read_img(content_img_path)
style_val = read_img(style_img_path)

content = tf.placeholder(tf.float32, shape=[1, 224, 224, 3])
style = tf.placeholder(tf.float32, shape=[1, 224, 224, 3])

data_dict = np.load(vgg16_npy_path, encoding='bytes').item()
vgg_for_content = VGGNet(data_dict)
vgg_for_style = VGGNet(data_dict)
vgg_for_result = VGGNet(data_dict)

vgg_for_content.build(content)  # VGGnet特征提取，每层卷积都是特征提取
vgg_for_style.build(style)      # 后面就可以算了
vgg_for_result.build(result)
# 内容是网络中越底层提取的特征越好,结果的内容和风格要和输入的内容风格对应
content_features = [
    # 深度学习中的卷积就是一种特征提取手段
    vgg_for_content.conv1_2,
    # vgg_for_content.conv2_2,
    # vgg_for_content.conv3_3,
    # vgg_for_content.conv4_3,
    # vgg_for_content.conv5_3,
]
result_content_features = [
    vgg_for_result.conv1_2,
    # vgg_for_result.conv2_2,
    # vgg_for_result.conv3_3,
    # vgg_for_result.conv4_3,
    # vgg_for_result.conv5_3,
]
# 风格是越高层越好
# feature_siez,[1, width, height, channel],每个channel是width和height大小
style_features = [
    # 深度学习中的卷积就是一种特征提取手段
    # vgg_for_style.conv1_2
    # vgg_for_style.conv2_2,
    # vgg_for_style.conv3_3,
    vgg_for_style.conv4_3,
    # vgg_for_style.conv5_3,
]
style_gram = [gram_matrix(feature) for feature in style_features]

result_style_features = [
    # vgg_for_result.conv1_2
    # vgg_for_result.conv2_2,
    # vgg_for_result.conv3_3,
    vgg_for_result.conv4_3,
    # vgg_for_result.conv5_3,
]
result_style_gram = [gram_matrix(feature) for feature in result_style_features]

# 内容损失，分别是各层损失的和
content_loss = tf.zeros(1, tf.float32)
# shape:[1， width， height, channel]
for c, c_ in zip(content_features, result_content_features):
    content_loss += tf.reduce_mean((c - c_) ** 2, [1, 2, 3])  # 后3个通道
# 风格损失 风格特征：[1, width, height, channel]
# 1.先算gram矩阵，它认为抽取出来的特征之间的关联性体现在channel中
# 每个channel都是width和height这样大小的图片(原图特征后的输出)
# 之后每个channel两两的就算相似度，(用余弦句柄)可以得到gram矩阵
# 2. 风格和结果的gram矩阵来计算风格损失
style_loss = tf.zeros(1, tf.float32)
for s, s_ in zip(style_gram, result_style_gram):
    style_loss += tf.reduce_mean((s - s_) ** 2, [1, 2])
loss = content_loss * lambda_c + style_loss * lambda_s
train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)

# 图像风格转换的流程
init_op = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init_op)
    for step in range(num_steps):
        loss_value, content_loss_value, style_loss_value, _ = \
            sess.run([loss, content_loss, style_loss, train_op],
                     feed_dict={
                         content: content_val,
                         style: style_val
                     })
        print('训练次数:%d, loss_value:%8.4f, content_loss:%8.4f, style_loss:%8.4f'\
              % (step+1, loss_value[0], content_loss_value[0], style_loss_value[0]))
        # 将每一次的图片存进文件
        result_img_path = os.path.join(output_dir, 'result-%05d.jpg' % (step+1))
        result_val = result.eval(sess)[0]  # 除了tf.Session().run()可以返回变量值,还可以直接eval()
        result_val = np.clip(result_val, 0, 255)  # 无法限制数字大小，只好裁剪
        img_arr = np.asarray(result_val, np.uint8)
        img = Image.fromarray(img_arr)
        img.save(result_img_path)

'''
图像风格转换V1的方式，需要每次都数据初化那张图片，之后用梯度下降的算法求解
为了得到比较的好结果，需要多次运行梯度下降算法，这样效率就会变低
V2进行改进，就是初化化的图片不在随机，而是经过一个网络（针对某个风格建立的网络）得到
V1,V2的损失函数计算方式都是一样，通过预训练好的VGG16。

图像超清化：
在V2的模型上，不要风格特性及其损失，输入的低分辨率，经过网络（低分辨到高分辨的一个映射），
得到的图片和输入低分辨的高分辩图像，经过VGG16计算两者损失。
'''