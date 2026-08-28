# 訓練
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import matplotlib.pyplot as plt
import time
import tqdm
import random
import numpy as np
from torch import nn, optim
import torch
from torch.utils.data import DataLoader, random_split
from data import *
from Unet import UNet




def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True   
    torch.backends.cudnn.benchmark = False      
 
set_seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#  你的檔案與輸出設定
weight_paths = [''] # 權重檔名.pth
data_path = '' # 資料路徑
save_paths = ['/train_image_lr_0.001'] # 儲存路徑

if __name__ == '__main__':
    num_classes = 1 + 1  # +1 是背景也為一類
    data_loader = DataLoader(MyDataset(data_path), batch_size=4, shuffle=True)

    #  動態偵測 DataLoader 實際輸入尺寸 (H, W)
    try:
        _sample_images, _sample_labels = next(iter(data_loader))
        inH, inW = _sample_images.shape[-2], _sample_images.shape[-1]
    except Exception as e:
        inH, inW = -1, -1  # 若失敗就保留 -1 作為未知
        print(f" 無法自動偵測輸入尺寸：{e}")

    # 定義學習率列表（可多個）
    learning_rates = [0.001]

    # 以學習率為 key，存每個學習率對應的 loss 與時間
    lr_losses = {lr: [] for lr in learning_rates}
    lr_training_times = {lr: 0 for lr in learning_rates}

    for lr, weight_path, save_path in zip(learning_rates, weight_paths, save_paths):

        # ========== 選擇模型 ==========
        # 引入模型
        #net = UNet(num_classes).to(device)

        #  自動抓模型名稱
        model_name = net.__class__.__name__

        print("=" * 60)
        print(f" PyTorch 版本：{torch.__version__}")
        print(f" CUDA 是否可用：{torch.cuda.is_available()}")
        print(f" 使用的裝置：{torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
        print(f" Model: {model_name}")
        print(f"  Input size (HxW): {inH} x {inW}  (由 DataLoader 實際偵測)")
        print("=" * 60)


        # 載入預訓練權重（若存在）
        if os.path.exists(weight_path):
            try:
                net.load_state_dict(torch.load(weight_path, map_location=device))
                print('成功加載權重！')
            except Exception as e:
                print(f'加載權重失敗：{e}')
        else:
            print('找不到指定權重檔，將初始化開始訓練。')

        # Optimizer / Loss
        opt = optim.Adam(net.parameters(), lr=lr)
        loss_fun = nn.CrossEntropyLoss()

        start_time = time.time()

        # ========== 訓練迴圈 ==========
        for epoch in range(1, 201):  # 從 1 到 200，共 200 次訓練
            net.train()
            total_loss = 0.0  # 累計 epoch loss

            for i, (image, segment_image) in enumerate(tqdm.tqdm(data_loader)):
                image, segment_image = image.to(device), segment_image.to(device)
                out_image = net(image)  # (N, C, H, W)
                train_loss = loss_fun(out_image, segment_image.long())

                opt.zero_grad()
                train_loss.backward()
                opt.step()

                total_loss += train_loss.item()

            average_loss = total_loss / len(data_loader)
            print(f'{epoch}-average_train_loss (lr={lr}) ===>> {average_loss}')

            # 記錄學習率和損失
            lr_losses[lr].append((epoch, average_loss))

            if epoch % 5 == 0:
                torch.save(net.state_dict(), weight_path)
                print('save successfully!')

        end_time = time.time()
        total_training_time = end_time - start_time
        lr_training_times[lr] = total_training_time

        # 訓練完後存 loss 到 CSV
        with open(f'unet_2800_train_loss_lr_{lr}.csv', 'w') as f:
            f.write('epoch,average_train_loss\n')
            for e, l in lr_losses[lr]:
                f.write(f'{e},{l}\n')


        # 在 while 循環結束後印出每個學習率的平均損失
        print(f"學習率 {lr} 的平均損失: {average_loss}")

    # 在所有學習率的訓練結束後計算和印出總平均損失
    for lr in learning_rates:
        total_average_loss = sum([loss for epoch, loss in lr_losses[lr]]) / len(lr_losses[lr])
        print(f"學習率 {lr} 的總平均損失: {total_average_loss}")

    # 繪製每個學習率的訓練損失圖
    colors = ['blue']
    for idx, lr in enumerate(learning_rates):
        losses = lr_losses[lr]
        epochs, losses = zip(*losses)
        plt.plot(epochs, losses, label=f'train loss', color=colors[idx])

    # 顯示每個學習率的訓練時間
    for lr in learning_rates:
        minutes = int(lr_training_times[lr] // 60)
        seconds = int(lr_training_times[lr] % 60)
        print(f"學習率 {lr} 的總訓練時間: {minutes} 分 {seconds} 秒")

    plt.xlabel('Epoch')
    plt.ylabel('Average Training Loss')
    plt.legend()
    plt.show()