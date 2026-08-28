# 測試單張圖評估指標
import torch
from PIL import Image
from torchvision import transforms
#from Unet import UNet
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import cv2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_classes = 2

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

def predict_single_image(image_path, model):
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        out_image = model(image)
        _, predicted = torch.max(out_image, 1)
    return predicted.squeeze().cpu().numpy()

def compute_iou(y_true, y_pred):
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    confusion = confusion_matrix(y_true, y_pred, labels=[0, 1])
    true_positive = confusion[1, 1]
    false_positive = confusion[0, 1]
    false_negative = confusion[1, 0]
    iou = true_positive / (true_positive + false_positive + false_negative) if (true_positive + false_positive + false_negative) > 0 else 0
    return iou

def draw_contour_on_image(image, mask, color=(255, 0, 0), thickness=1):
    """在原圖上畫出預測區域輪廓"""
    # image: PIL.Image or np.ndarray (H,W,3)
    # mask: np.ndarray (H,W), mask=1的地方畫出輪廓
    img = np.array(image.convert('RGB')) if isinstance(image, Image.Image) else image
    mask = (mask > 0).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_contour = img.copy()
    cv2.drawContours(img_contour, contours, -1, color, thickness)
    return img_contour

def test_single_image_all_models(image_path, label_path, weight_paths):
    models = {
        #"Unet":UNet(num_classes=2).to(device),
    }  
    for name, path in weight_paths.items():
        if path and name in models:
            models[name].load_state_dict(torch.load(path, map_location=device))

    label = Image.open(label_path)
    label = transforms.Resize((256, 256))(label)
    label = torch.Tensor(np.array(label))
    label = label.squeeze().numpy()
    y_true = label.flatten()

    orig_img = Image.open(image_path).resize((256, 256))

    for name, model in models.items():
        predicted = predict_single_image(image_path, model)
        y_pred = predicted.flatten()
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        iou = compute_iou(label, predicted)
        print(f"[{name}] Accuracy: {accuracy:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f} | IoU: {iou:.4f}")

        # 原圖畫上預測輪廓
        result_img = draw_contour_on_image(orig_img, predicted, color=(255, 0, 0), thickness=1)
        # Ground truth輪廓（綠色）
        gt_img = draw_contour_on_image(orig_img, label, color=(0, 255, 0), thickness=1)

        plt.figure(figsize=(12,4))
        plt.subplot(1, 2, 1)
        plt.imshow(gt_img)
        plt.title('GT Contour (Green)')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(result_img)
        plt.title(f'Pred Contour: {name} (Red)')
        plt.axis('off')

        plt.suptitle(name)
        plt.show()

if __name__ == '__main__':
    weight_paths = {
        #"Unet": "", # 權重路徑



    }

    test_single_image_all_models(
        "", # 要測試的圖像路徑
        "", # 要測試的圖像遮罩路徑
        weight_paths
    )