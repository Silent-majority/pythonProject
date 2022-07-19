import glob
import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from keras import layers, Sequential, optimizers

# 路径
path = 'data/flower_photos/'

# 高度宽度与RGB
h = 100
w = 100
c = 3


# 读取图片
def reg_img(path):
    print(os.listdir(path))
    imgs = []
    labels = []  # 文件夹索引作为label

    # 图片类别
    categories = [(path + x) for x in os.listdir(path)]
    print(categories)

    # 遍历文件夹，添加数据
    for index, floder in enumerate(categories):
        for im in glob.glob(floder + '/*.jpg'):
            # 读取图片
            img = cv2.imread(im)
            # 重置图片大小
            img = cv2.resize(img, (h, w))
            imgs.append(img)
            labels.append(index)

    # 转换为numpy数组
    return np.array(imgs, dtype='float32'), np.array(labels, dtype='int32')


def main():
    import tensorflow as tf
    tf.test.is_gpu_available()
    data, label = reg_img(path)

    # 数据集划分
    x_train, x_test, y_train, y_test = train_test_split(data, label, test_size=0.2, random_state=0)

    # 归一化处理
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    print(label)

    # 卷积神经网络
    model = Sequential([
        layers.Conv2D(32, 5, padding='same', activation=tf.nn.relu),
        layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'),

        # 随机丢弃
        layers.Dropout(0.2),

        layers.Conv2D(64, 3, padding='same', activation=tf.nn.relu),
        layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'),

        # 随机丢弃
        layers.Dropout(0.2),

        layers.Conv2D(128, 3, padding='same', activation=tf.nn.relu),
        layers.Conv2D(128, 3, padding='same', activation=tf.nn.relu),
        layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'),

        # 随机丢弃
        layers.Dropout(0.2),

        # 将每一次的输出展平
        layers.Flatten(),

        # 全连接层
        layers.Dense(512, activation=tf.nn.relu),
        layers.Dense(256, activation=tf.nn.relu),

        # 输出层
        layers.Dense(10, activation=tf.nn.softmax)
    ])

    # 编译模型
    model.compile(optimizer=optimizers.Adam(lr=0.001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # 训练模型
    model.fit(x_train, y_train, epochs=20, validation_data=(x_test, y_test), batch_size=128)

    # 测试模型
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print('test_acc:', test_acc)

    # 保存模型
    model.save('model.h5')

    # 加载模型
    model = tf.keras.models.load_model('model.h5')

    # 预测
    pred = model.predict(x_test)
    print(pred)

    # 模型评估
    print(model.evaluate(x_test, y_test))


if __name__ == '__main__':
    main()
