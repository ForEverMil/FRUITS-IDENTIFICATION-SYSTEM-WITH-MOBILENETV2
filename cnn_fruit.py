#import pandas as pd
#import plotly.express as px
#import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, AveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras import regularizers
from tensorflow.keras.regularizers import l1
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.applications import MobileNetV2
import os
#import matplotlib.pyplot as plt
import random
import numpy as np
from pathlib import Path
from keras.preprocessing.image import load_img
import cv2

#from sklearn.preprocessing import LabelEncoder

import warnings
warnings.filterwarnings('ignore')


# Tạo các đối tượng ImageDataGenerator cho việc augmentation dữ liệu huấn luyện, validation và kiểm tra
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    validation_split=0.2
)

val_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

test_datagen = ImageDataGenerator(rescale=1./255)

# Tạo các dữ liệu generator từ thư mục chứa dữ liệu ảnh
train_ds = train_datagen.flow_from_directory(
    directory='C:\\Users\\hnc1411\\PycharmProjects\\nhandientraicay\\.venv\\data\\train',
    batch_size=32,
    target_size=(224, 224),
    class_mode='categorical',
    subset="training",
    seed=123
)

validation_ds = val_datagen.flow_from_directory(
    directory='C:\\Users\\hnc1411\\PycharmProjects\\nhandientraicay\\.venv\\data\\train',
    batch_size=32,
    target_size=(224, 224),
    class_mode='categorical',
    subset="validation",
    seed=123
)

test_ds = train_datagen.flow_from_directory(
    directory='C:\\Users\\hnc1411\\PycharmProjects\\nhandientraicay\\.venv\\data\\test',
    batch_size=32,
    target_size=(224, 224),
    class_mode='categorical'
)
"""
def visualize_images(path, num_images=5):

    # Get a list of image filenames
    image_filenames = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]

    if not image_filenames:
        raise ValueError("No images found in the specified path")

    # Select random images
    selected_images = random.sample(image_filenames, min(num_images, len(image_filenames)))

    # Create a figure and axes
    fig, axes = plt.subplots(1, num_images, figsize=(15, 3), facecolor='white')

    # Display each image
    for i, image_filename in enumerate(selected_images):
        # Load image
        image_path = os.path.join(path, image_filename)
        image = plt.imread(image_path)

        # Display image
        axes[i].imshow(image)
        axes[i].axis('off')
        axes[i].set_title(image_filename)  # Set image filename as title

    # Adjust layout and display
    plt.tight_layout()
    plt.show()

    # Specify the path containing the images to visualize
    path_to_visualize = "C:\\Users\\hnc1411\\PycharmProjects\\nhandientraicay\\.venv\\data\\train\\Chuối"

    visualize_images(path_to_visualize, num_images=5)

    # Specify the path containing the images to visualize
    path_to_visualize = "C:\\Users\\hnc1411\\PycharmProjects\\nhandientraicay\\.venv\\data\\train\\Dâu tây"

    visualize_images(path_to_visualize, num_images=5)

    # Specify the path containing the images to visualize
    path_to_visualize = "C:\\Users\\hnc1411\\PycharmProjects\\nhandientraicay\\.venv\\data\\train\\Bơ"

    visualize_images(path_to_visualize, num_images=5)

    # Specify the path containing the images to visualize
    path_to_visualize = "C:\\Users\\hnc1411\\PycharmProjects\\nhandientraicay\\.venv\\data\\train\\Dứa"

    visualize_images(path_to_visualize, num_images=5)

    # Specify the path containing the images to visualize
    path_to_visualize = "C:\\Users\\hnc1411\\PycharmProjects\\nhandientraicay\\.venv\\data\\train\\Dưa hấu"

    visualize_images(path_to_visualize, num_images=5)

    # Specify the path containing the images to visualize
    path_to_visualize = "C:\\Users\\hnc1411\\PycharmProjects\\nhandientraicay\\.venv\\data\\train\\Kiwi"

    visualize_images(path_to_visualize, num_images=5)

    # Specify the path containing the images to visualize
    path_to_visualize = "C:\\Users\\hnc1411\\PycharmProjects\\nhandientraicay\\.venv\\data\\train\\Cherry"

    visualize_images(path_to_visualize, num_images=5)


"""
# Tải mô hình MobileNetV2 đã được huấn luyện trước mà không có lớp phân loại đầu ra
MobileNetV2_base = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3), pooling='avg')

# Đóng băng các lớp của mô hình cơ sở đã được huấn luyện
MobileNetV2_base.trainable = False

# Xây dựng mô hình
model = Sequential()

# Thêm mô hình cơ sở MobileNetV2 đã được huấn luyện trước
model.add(MobileNetV2_base)

# Chuẩn hóa theo batch
model.add(BatchNormalization())

# Lớp dropout
model.add(Dropout(0.35))

# Thêm một lớp dense với 220 đơn vị và hàm kích hoạt ReLU
model.add(Dense(220, activation='relu'))

# Thêm một lớp dense với 60 đơn vị và hàm kích hoạt ReLU
model.add(Dense(60, activation='relu'))

# Thêm lớp đầu ra với 10 đơn vị và hàm kích hoạt softmax
model.add(Dense(10, activation='softmax'))

# Biên dịch mô hình
base_learning_rate = 0.0001
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
              loss='categorical_crossentropy', metrics=['accuracy'])

# Xác định hàm gọi lại
early_stopping = EarlyStopping(patience=10)

# Huấn luyện mô hình
history = model.fit(train_ds,
                    validation_data=validation_ds,
                    steps_per_epoch=len(train_ds),
                    epochs=50,
                    callbacks=[early_stopping])

# Lưu mô hình
model.save('model.h5')
"""
# Đánh giá mô hình
loss = model.evaluate(validation_ds)

# Vẽ đồ thị mất mát huấn luyện và kiểm tra
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Mất mát của mô hình')
plt.ylabel('Mất mát')
plt.xlabel('Epoch')
plt.legend(['Huấn luyện', 'Kiểm tra'], loc='upper right')
plt.show()

# Vẽ đồ thị độ chính xác của huấn luyện và kiểm tra
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Độ chính xác của mô hình')
plt.ylabel('Độ chính xác')
plt.xlabel('Epoch')
plt.legend(['Huấn luyện', 'Kiểm tra'], loc='lower right')
plt.show()

# Lấy nhãn lớp
class_labels = list(test_ds.class_indices.keys())

# Dự đoán trên mỗi hình ảnh và vẽ kết quả
num_images = 20
num_images_per_row = 5  # Số lượng hình ảnh mỗi hàng
num_rows = 4

plt.figure(figsize=(15, 10))
for i in range(num_images):
    image, label = next(test_ds)
    predictions = model.predict(image)

    # Lặp qua mỗi hình ảnh trong batch
    for j in range(len(image)):
        if i * len(image) + j < num_images:  # Kiểm tra xem tổng số hình ảnh có vượt quá số lượng mong muốn không
            predicted_class = class_labels[np.argmax(predictions[j])]
            true_class = class_labels[np.argmax(label[j])]

            plt.subplot(num_rows, num_images_per_row, i * len(image) + j + 1)
            plt.imshow(image[j])
            plt.title(f'Chuẩn xác: {true_class}\nDự đoán: {predicted_class}')
            plt.axis('off')

plt.tight_layout()
plt.show()
"""