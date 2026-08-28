# 測試模型評估指標
import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, jaccard_score

# ==== 匯入模型 ==== #
#from Unet import UNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

def dice_score(y_true, y_pred):
    smooth = 1e-6
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = (y_true_f * y_pred_f).sum()
    return (2. * intersection + smooth) / (y_true_f.sum() + y_pred_f.sum() + smooth)

def predict_mask(model, img_path):
    image = Image.open(img_path).convert('RGB')
    img_tensor = transform(image).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        output = model(img_tensor)
        pred = torch.argmax(output, dim=1)
    return pred.squeeze().cpu().numpy()

def compute_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    iou = jaccard_score(y_true, y_pred, average='binary', zero_division=0)
    dice = dice_score(y_true, y_pred)
    return acc, prec, rec, f1, iou, dice

def evaluate_model(model, weight_path, image_dir, mask_dir):
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.to(device)
    all_preds, all_gts = [], []

    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))])
    for fname in image_files:
        img_path = os.path.join(image_dir, fname)
        mask_path = os.path.join(mask_dir, os.path.splitext(fname)[0] + ".png")
        if not os.path.exists(mask_path): continue

        gt = np.array(Image.open(mask_path).resize((256, 256)))
        pred = predict_mask(model, img_path)
        all_gts.append(gt.flatten())
        all_preds.append(pred.flatten())

    y_true = np.concatenate(all_gts)
    y_pred = np.concatenate(all_preds)
    return compute_metrics(y_true, y_pred)

if __name__ == "__main__":
    # 資料路徑
    image_dir = ""
    mask_dir  = ""

    # 模型與權重
    models = {
        #"Unet": (UNet(num_classes=2), "C:/Users/user/Desktop/pytorch-UNet-master-1/pytorch-UNet-master/params/unet_2800.pth"),
    }
    

    print("===  模型評估指標比較 ===")
    for name, (model, weight) in models.items():
        acc, prec, rec, f1, iou, dice = evaluate_model(model, weight, image_dir, mask_dir)
        print(f"【{name}】")
        print(f"  Accuracy : {acc:.4f}")
        print(f"  Precision: {prec:.4f}")
        print(f"  Recall   : {rec:.4f}")
        print(f"  F1 Score : {f1:.4f}")
        print(f"  IoU      : {iou:.4f}")
        print(f"  Dice     : {dice:.4f}")
        print("-" * 30)