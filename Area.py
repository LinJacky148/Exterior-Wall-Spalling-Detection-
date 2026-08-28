# 計算面積
import torch
from PIL import Image
from torchvision import transforms

#from Unet import UNet


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

import numpy as np
import matplotlib.pyplot as plt
import cv2


# ============================================================
# 基本設定
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 2
IMAGE_SIZE = (256, 256)

# 每塊磁磚實際尺寸為 10 × 10 cm，即 0.1 × 0.1 m
TILE_WIDTH_M = 0.1
TILE_HEIGHT_M = 0.1
TILE_AREA_M2 = TILE_WIDTH_M * TILE_HEIGHT_M

# 固定隨機性
torch.manual_seed(42)
np.random.seed(42)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor()
])


# ============================================================
# 單張影像預測
# ============================================================

def predict_single_image(image_path, model):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    model.eval()

    with torch.no_grad():
        output = model(image)
        predicted = torch.argmax(output, dim=1)

    return predicted.squeeze(0).cpu().numpy().astype(np.uint8)


# ============================================================
# 計算 IoU
# ============================================================

def compute_iou(y_true, y_pred):
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    confusion = confusion_matrix(y_true, y_pred, labels=[0, 1])

    true_positive = confusion[1, 1]
    false_positive = confusion[0, 1]
    false_negative = confusion[1, 0]

    denominator = true_positive + false_positive + false_negative

    return true_positive / denominator if denominator > 0 else 0.0


# ============================================================
# 在原圖上繪製遮罩輪廓
# ============================================================

def draw_contour_on_image(image, mask, color=(255, 0, 0), thickness=1):
    image_array = np.array(image.convert("RGB")) if isinstance(image, Image.Image) else image.copy()
    binary_mask = (mask > 0).astype(np.uint8)

    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    image_contour = image_array.copy()
    cv2.drawContours(image_contour, contours, -1, color, thickness)

    return image_contour


# ============================================================
# 計算遮罩實際面積，單位為 m²
# ============================================================

def calculate_mask_area(mask, total_reference_area_m2):
    binary_mask = (mask > 0).astype(np.uint8)

    pixel_count = int(np.sum(binary_mask))
    total_pixel_count = binary_mask.size

    area_ratio = pixel_count / total_pixel_count
    area_m2 = area_ratio * total_reference_area_m2

    return pixel_count, area_ratio, area_m2


# ============================================================
# 測試單張影像
# ============================================================

def test_single_image_all_models(image_path, label_path, weight_paths):
    models = {
        #"Unet": UNet(num_classes=2).to(device),  # 引入模型
 
    }

    # 載入各模型權重
    for name, path in weight_paths.items():
        if path and name in models:
            models[name].load_state_dict(torch.load(path, map_location=device))
            print(f"已載入模型權重：{name}")

    # 輸入影像中的完整磁磚排列數量
    tile_columns = int(input("請輸入橫向磁磚格數："))
    tile_rows = int(input("請輸入縱向磁磚格數："))

    if tile_columns <= 0 or tile_rows <= 0:
        raise ValueError("磁磚橫向與縱向格數必須大於 0。")

    # 實際剝落磁磚數量只用於計算實體面積誤差
    removed_tile_input = input("請輸入實際剝落磁磚數量（未知可直接按 Enter）：").strip()
    removed_tile_count = float(removed_tile_input) if removed_tile_input else None

    total_tile_positions = tile_columns * tile_rows

    if removed_tile_count is not None and (removed_tile_count < 0 or removed_tile_count > total_tile_positions):
        raise ValueError("實際剝落磁磚數量不可小於 0 或大於磁磚位置總數。")

    # 計算整體磁磚排列面積，單位為 m²
    total_reference_area_m2 = total_tile_positions * TILE_AREA_M2

    # 計算實體標準剝落面積，單位為 m²
    physical_gt_area_m2 = removed_tile_count * TILE_AREA_M2 if removed_tile_count is not None else None

    print("\n============================================================")
    print(f"使用裝置：{device}")
    print(f"影像尺寸：{IMAGE_SIZE[0]} × {IMAGE_SIZE[1]}")
    print(f"磁磚排列：{tile_columns} 欄 × {tile_rows} 列")
    print(f"磁磚位置總數：{total_tile_positions}")
    print(f"單塊磁磚面積：{TILE_AREA_M2:.4f} m²")
    print(f"磁磚排列總面積：{total_reference_area_m2:.4f} m²")

    if physical_gt_area_m2 is not None:
        print(f"實際剝落磁磚數量：{removed_tile_count:g} 塊")
        print(f"實體標準剝落面積：{physical_gt_area_m2:.6f} m²")

    print("============================================================")

    # 讀取人工標註遮罩
    label_image = Image.open(label_path).convert("L")
    label_image = transforms.Resize(IMAGE_SIZE, interpolation=transforms.InterpolationMode.NEAREST)(label_image)
    label = (np.array(label_image) > 0).astype(np.uint8)

    y_true = label.flatten()

    # 讀取原始影像
    orig_img = Image.open(image_path).convert("RGB").resize(IMAGE_SIZE)

    # 人工標註遮罩面積
    gt_pixel_count, gt_area_ratio, gt_mask_area_m2 = calculate_mask_area(label, total_reference_area_m2)

    print("\n============================================================")
    print("人工標註遮罩結果")
    print("============================================================")
    print(f"標註剝落像素數：{gt_pixel_count} pixels")
    print(f"標註剝落比例：{gt_area_ratio * 100:.2f}%")
    print(f"標註遮罩換算面積：{gt_mask_area_m2:.6f} m²")

    if physical_gt_area_m2 is not None:
        gt_physical_absolute_error = abs(gt_mask_area_m2 - physical_gt_area_m2)
        gt_physical_relative_error = gt_physical_absolute_error / physical_gt_area_m2 * 100 if physical_gt_area_m2 > 0 else 0.0

        print(f"實體標準面積：{physical_gt_area_m2:.6f} m²")
        print(f"標註遮罩與實體面積差異：{gt_physical_absolute_error:.6f} m²")
        print(f"標註遮罩與實體面積誤差：{gt_physical_relative_error:.2f}%")

    print("============================================================")

    # ========================================================
    # 各模型預測與面積計算
    # ========================================================

    for name, model in models.items():
        predicted = predict_single_image(image_path, model)
        predicted = (predicted > 0).astype(np.uint8)

        y_pred = predicted.flatten()

        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        iou = compute_iou(label, predicted)

        # 模型預測遮罩面積，單位為 m²
        pred_pixel_count, pred_area_ratio, pred_area_m2 = calculate_mask_area(predicted, total_reference_area_m2)

        # 模型相對於人工標註遮罩的面積誤差
        mask_absolute_error = abs(pred_area_m2 - gt_mask_area_m2)
        mask_relative_error = mask_absolute_error / gt_mask_area_m2 * 100 if gt_mask_area_m2 > 0 else 0.0

        # 模型相對於實體標準答案的面積誤差
        if physical_gt_area_m2 is not None:
            physical_absolute_error = abs(pred_area_m2 - physical_gt_area_m2)
            physical_relative_error = physical_absolute_error / physical_gt_area_m2 * 100 if physical_gt_area_m2 > 0 else 0.0
        else:
            physical_absolute_error = None
            physical_relative_error = None

        print(f"\n[{name}]")
        print(f"Accuracy: {accuracy:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f} | IoU: {iou:.4f}")
        print(f"人工標註遮罩面積：{gt_mask_area_m2:.6f} m²")
        print(f"模型預測剝落像素數：{pred_pixel_count} pixels")
        print(f"模型預測剝落比例：{pred_area_ratio * 100:.2f}%")
        print(f"模型預測剝落面積：{pred_area_m2:.6f} m²")
        print(f"相對人工標註絕對誤差：{mask_absolute_error:.6f} m²")
        print(f"相對人工標註面積誤差：{mask_relative_error:.2f}%")

        if physical_gt_area_m2 is not None:
            print(f"實體標準剝落面積：{physical_gt_area_m2:.6f} m²")
            print(f"相對實體標準絕對誤差：{physical_absolute_error:.6f} m²")
            print(f"相對實體標準面積誤差：{physical_relative_error:.2f}%")

        # 繪製人工標註與模型預測輪廓
        gt_img = draw_contour_on_image(orig_img, label, color=(0, 255, 0), thickness=1)
        result_img = draw_contour_on_image(orig_img, predicted, color=(255, 0, 0), thickness=1)

        if physical_gt_area_m2 is not None:
            gt_title = f"Physical GT: {physical_gt_area_m2:.6f} m²\nAnnotated Mask: {gt_mask_area_m2:.6f} m²"
            pred_title = f"Predicted: {pred_area_m2:.6f} m²\nPhysical Error: {physical_relative_error:.2f}% | Mask Error: {mask_relative_error:.2f}%"
        else:
            gt_title = f"Annotated Mask: {gt_mask_area_m2:.6f} m²"
            pred_title = f"Predicted: {pred_area_m2:.6f} m²\nMask Error: {mask_relative_error:.2f}%"

        plt.figure(figsize=(13, 5))

        plt.subplot(1, 2, 1)
        plt.imshow(gt_img)
        plt.title(gt_title)
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(result_img)
        plt.title(pred_title)
        plt.axis("off")

        plt.suptitle(f"{name} | IoU: {iou:.4f} | F1: {f1:.4f}")
        plt.tight_layout()
        plt.show()


# ============================================================
# 執行程式
# ============================================================

if __name__ == "__main__":
    weight_paths = {
        "Unet": "", # 權重路徑
    }

    test_single_image_all_models(
        "",
        "",
        weight_paths
    )