# 訓練姿勢模型（選用、僅個人使用）

模型輸入頭顯、雙手、雙小腿方向，輸出臀部與膝蓋／腳踝位置建議。架構參考 HMD-Poser（CVPR 2024，支援「頭顯 + 雙小腿 IMU」配置）：每幀嵌入 + LSTM，可在 CPU 即時執行。

## 1. 下載資料（需註冊，非商業研究授權）

- AMASS：https://amass.is.tue.mpg.de → Downloads → 選 **SMPL+H G** 格式的資料集（建議至少 CMU、BMLmovi、KIT、HDM05、BMLrub、TotalCapture），解壓到 `data\amass\`
- SMPL+H 模型：同一網站的「Extended SMPL+H model」→ 解壓成 `data\smplh\neutral\model.npz`（以及 male／female）

## 2. 安裝訓練套件

```
.venv\Scripts\python.exe -m pip install -r training\requirements.txt
```

有 NVIDIA 顯示卡時，依 https://pytorch.org 安裝 CUDA 版 torch 會快很多。

## 3. 前處理 → 訓練 → 匯出

```
.venv\Scripts\python.exe training\prepare_amass.py --amass data\amass --smplh data\smplh --out data\train\amass60.npz
.venv\Scripts\python.exe training\train.py --data data\train\amass60.npz --out models\pose.pt --epochs 40
.venv\Scripts\python.exe training\export_onnx.py --ckpt models\pose.pt --out models\pose.onnx
```

訓練時會模擬 AirPods 的誤差：每條腿的殘留 yaw 偏移（約 ±8°）、緩慢飄移、綁帶角度誤差、雜訊，以及手把短暫追蹤丟失。

## 授權

AMASS 與 SMPL+H 只允許非商業研究使用，且禁止散布資料；用它們訓練出的權重 `pose.pt` / `pose.onnx` 請只留在自己電腦，不要上傳或分享。
