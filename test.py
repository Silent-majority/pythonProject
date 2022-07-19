import glob

import cv2
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras


def test():
    # 加载模型
    model = keras.models.load_model('model.h5')

    test_path = 'data/TestImages/'

    test_images = []

    for im in glob.glob(test_path + '*.jpg'):
        img = cv2.imread(im)
        img = cv2.resize(img, (100, 100))
        test_images.append(img)

    test_images = np.array(test_images, dtype='float32')

    # 模型预测
    predictions = model.predict(test_images)
    print(predictions)

    dir = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']
    for index in range(5):
        img = plt.imread(test_path + "test" + str(index) + ".jpg")
        print("预测结果：", dir[predictions[index].tolist().index(1.0)])
        plt.imshow(img)
        plt.show()


test()
