# 分割出標準答案
import cv2
import numpy as np
import os

# 設定資料夾路徑
image_folder = "" # 圖像路徑
mask_folder = "C:/Users/user/Desktop/pytorch-UNet-master-1/pytorch-UNet-master/data/TestSegmentationClass.1.256" # 圖像遮罩路徑
output_folder = "C:/Users/user/Desktop/pytorch-UNet-master-1/pytorch-UNet-master/data/testanswer256" # 要儲存的路徑

# 建立輸出資料夾
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 遍歷原圖資料夾
for image_file in os.listdir(image_folder):

    # 只處理 jpg / jpeg
    if not image_file.lower().endswith((".jpg", ".jpeg")):
        continue

    image_path = os.path.join(image_folder, image_file)

    # 取得不含副檔名的檔名
    file_name = os.path.splitext(image_file)[0]

    # mask 是 png
    mask_file = file_name + ".png"
    mask_path = os.path.join(mask_folder, mask_file)

    # 檢查 mask 是否存在
    if not os.path.exists(mask_path):
        print(f"Mask file {mask_file} not found, skipping...")
        continue

    # 讀取原圖與 mask
    original_image = cv2.imread(image_path)
    mask_image = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if original_image is None:
        print(f"Image {image_file} read failed, skipping...")
        continue

    if mask_image is None:
        print(f"Mask {mask_file} read failed, skipping...")
        continue

    # 檢查 mask 數值，確認是不是 0 / 1
    print(f"{mask_file} unique values:", np.unique(mask_image))

    # 因為 mask 是 0 / 1，所以把 1 轉成 255
    mask_image = np.where(mask_image > 0, 255, 0).astype(np.uint8)

    # 如果原圖和 mask 尺寸不同，將 mask resize 成原圖大小
    if original_image.shape[:2] != mask_image.shape[:2]:
        mask_image = cv2.resize(
            mask_image,
            (original_image.shape[1], original_image.shape[0]),
            interpolation=cv2.INTER_NEAREST
        )

    # 找 mask 輪廓
    contours, _ = cv2.findContours(
        mask_image,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    print(f"{image_file} contours found: {len(contours)}")

    # 複製原圖並畫上綠色輪廓
    contour_image = original_image.copy()
    cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 1)

    # 輸出仍然存成 jpg
    output_path = os.path.join(output_folder, image_file)
    cv2.imwrite(output_path, contour_image)

print("所有圖像處理完畢，輸出圖像已保存。")