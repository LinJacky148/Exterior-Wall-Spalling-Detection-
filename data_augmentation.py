# 隨機旋轉正負30度
import os
import cv2
import numpy as np

def random_rotate_images_in_directory(directory_path, save_path, max_angle=30):
    # 確保保存路徑存在
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # 列出資料夾中的所有圖像文件
    image_files = [f for f in os.listdir(directory_path) if f.endswith(('.jpg', '.jpeg', '.png', '.bmp'))]

    for filename in image_files:
        # 讀取圖像
        image_path = os.path.join(directory_path, filename)
        image = cv2.imread(image_path)

        # 隨機生成旋轉角度
        angle = np.random.uniform(-max_angle, max_angle)

        # 獲取圖像尺寸
        rows, cols, _ = image.shape
        # 計算旋轉中心
        center = (cols // 2, rows // 2)
        # 獲取旋轉矩陣
        M = cv2.getRotationMatrix2D(center, angle, 1)
        # 執行旋轉
        rotated_image = cv2.warpAffine(image, M, (cols, rows))

        # 保存旋轉後的圖像
        save_name = os.path.join(save_path, filename.split('.')[0] + '_rotated' + '.' + filename.split('.')[1])
        cv2.imwrite(save_name, rotated_image)

# 資料夾路徑和保存路徑
input_directory = 'C:/Users/user/Desktop/pytorch-UNet-master-1/pytorch-UNet-master/data/JPEGImages'  # 替換成你的資料夾路徑
output_directory = 'C:/Users/user/Desktop/NewJPEGImages'  # 替換成你的保存路徑

# 執行隨機旋轉
random_rotate_images_in_directory(input_directory, output_directory)