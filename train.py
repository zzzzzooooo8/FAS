import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import os
import copy
import pandas as pd 
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import cv2 

# ==========================================
# 1. 数据集定义 (已升级为：下肢专注模式)
# ==========================================
class FitnessDataset(Dataset):
    def __init__(self, root_dir, augment=True):
        self.root_dir = root_dir
        self.classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.samples = []
        self.augment = augment
        
        # 翻转对
        self.flip_pairs = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)]

        for cls_name in self.classes:
            cls_path = os.path.join(root_dir, cls_name)
            file_names = [f for f in os.listdir(cls_path) if f.endswith('.npy')]
            for file_name in file_names:
                self.samples.append((os.path.join(cls_path, file_name), self.class_to_idx[cls_name], False))
                if augment:
                    self.samples.append((os.path.join(cls_path, file_name), self.class_to_idx[cls_name], True))
        
        print(f"数据加载完毕: 共 {len(self.classes)} 类, 样本总数 {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label, is_flip = self.samples[idx]
        data = np.load(path) 
        
        if is_flip: data = self._flip_data(data)
        
        if self.augment:
            speed_factor = np.random.choice([0.8, 0.9, 1.0, 1.1, 1.2]) 
            data = self._apply_speed_jitter(data, speed_factor)
            data = self._mask_body_parts(data)
            temp = data.reshape(data.shape[0], 17, 3)
            noise = np.random.normal(0, 0.002, temp[:, :, :2].shape) # 噪声调小一点，防止把关键特征淹没
            temp[:, :, :2] += noise
            data = temp.reshape(data.shape[0], -1)

        # 1. 屏蔽干扰部位 (脸、手臂)
        data = self._filter_noise_parts(data)

        # 2. 归一化 (中心化 + 尺度缩放) -> 核心修复
        data = self._normalize_pose(data)
        
        data_tensor = torch.from_numpy(data).float()
        return data_tensor, label

    def _filter_noise_parts(self, data):
        """
        屏蔽五官(0-4)和手臂(7-10)，只保留躯干(5,6,11,12)和下肢(13-16)
        """
        frames = data.reshape(data.shape[0], 17, 3)
        ignore_indices = [0, 1, 2, 3, 4, 7, 8, 9, 10]
        frames[:, ignore_indices, :] = 0.0
        return frames.reshape(data.shape[0], -1)

    def _normalize_pose(self, data):
        """
        【分级锚点 + 尺度归一化】
        解决"站得远"和"蹲得近"分不清的问题
        """
        frames = data.reshape(data.shape[0], 17, 3)
        
        for i in range(frames.shape[0]):
            frame = frames[i]
            confs = frame[:, 2]
            
            # --- 1. 确定原点 (Anchor) ---
            anchor = np.zeros(2)
            has_anchor = False
            
            # 优先用髋部
            if confs[11] > 0.1 and confs[12] > 0.1:
                anchor = (frame[11, :2] + frame[12, :2]) / 2
                has_anchor = True
            # 次选用肩膀
            elif confs[5] > 0.1 and confs[6] > 0.1:
                anchor = (frame[5, :2] + frame[6, :2]) / 2
                has_anchor = True
            # 兜底
            elif np.any(confs > 0):
                anchor = np.mean(frame[confs > 0, :2], axis=0)
                has_anchor = True

            # --- 2. 确定缩放比例 (Scale) ---
            # 目标：把躯干长度归一化为 1.0 左右
            scale = 1.0
            # 只有当肩膀和屁股都看得到时，才能算躯干长度
            if confs[5] > 0.1 and confs[6] > 0.1 and confs[11] > 0.1 and confs[12] > 0.1:
                # 肩膀中心
                shoulder_center = (frame[5, :2] + frame[6, :2]) / 2
                # 髋部中心
                hip_center = (frame[11, :2] + frame[12, :2]) / 2
                
                # 计算垂直距离 (或欧氏距离)，作为标尺
                # 这里用欧氏距离更稳，防止侧身时垂直距离变小
                torso_len = np.linalg.norm(shoulder_center - hip_center)
                
                if torso_len > 0.05: # 防止除以0
                    scale = torso_len

            # --- 3. 执行变换 ---
            if has_anchor:
                frame[:, :2] -= anchor # 平移
                frame[:, :2] /= scale  # 缩放 (让所有人变成一样大)
            
            # 清理无效点
            frame[~(confs > 0), :2] = 0

        return frames.reshape(data.shape[0], -1)

    # 辅助函数保持不变
    def _flip_data(self, data):
        data = data.reshape(data.shape[0], 17, 3)
        data[:, :, 0] = 1.0 - data[:, :, 0] 
        for i, j in self.flip_pairs:
            temp = data[:, i, :].copy() 
            data[:, i, :] = data[:, j, :]
            data[:, j, :] = temp
        return data.reshape(data.shape[0], -1)

    def _mask_body_parts(self, data):
        rand = np.random.random()
        temp = data.reshape(data.shape[0], 17, 3)
        if rand < 0.15: temp[:, 11:17, :] = 0.0 
        elif rand < 0.3: temp[:, 0:11, :] = 0.0
        return temp.reshape(data.shape[0], -1)

    def _apply_speed_jitter(self, data, speed_factor):
        seq_len, num_features = data.shape 
        virtual_len = max(1, int(seq_len * speed_factor))
        resampled = cv2.resize(data, (num_features, virtual_len), interpolation=cv2.INTER_LINEAR)
        output = cv2.resize(resampled, (num_features, seq_len), interpolation=cv2.INTER_LINEAR)
        return output

    # --- 新增函数：下肢专注过滤器 ---
    def _filter_lower_body(self, data):
        """
        强行将上半身非核心区域置 0
        保留: 肩(5,6), 髋(11,12), 膝(13,14), 踝(15,16)
        屏蔽: 五官(0-4), 肘(7,8), 腕(9,10)
        """
        frames = data.reshape(data.shape[0], 17, 3)
        # 定义要屏蔽的索引
        ignore_indices = [0, 1, 2, 3, 4, 7, 8, 9, 10]
        # 简单粗暴：坐标和置信度全部变 0
        frames[:, ignore_indices, :] = 0.0
        return frames.reshape(data.shape[0], -1)

     # === 【修改点 2】 新增分级锚点函数 (替换了原来的 _center_data_universal) ===

    def _center_data_hierarchical(self, data):

        """
        【通用分级锚点策略】
        1. 第一优先级：髋关节中心 (Hips: 11, 12) -> 最稳，适合全身动作，抗手部干扰
        2. 第二优先级：肩关节中心 (Shoulders: 5, 6) -> 适合半身动作
        3. 第三优先级：可见点重心 (All Visible) -> 兜底策略
        """
        frames = data.reshape(data.shape[0], 17, 3)
        for i in range(frames.shape[0]):
            frame = frames[i] # (17, 3)
            confs = frame[:, 2]
            # --- 分级寻找原点 ---
            anchor_point = None
            # 1. 尝试用髋部 (11, 12)
            if confs[11] > 0 and confs[12] > 0:
                anchor_point = (frame[11, :2] + frame[12, :2]) / 2
            # 2. 如果没髋部，尝试用肩膀 (5, 6)
            elif confs[5] > 0 and confs[6] > 0:
                anchor_point = (frame[5, :2] + frame[6, :2]) / 2
            # 3. 如果都没，用所有可见点的重心兜底
            else:
                valid_mask = confs > 0
                if np.any(valid_mask):
                    anchor_point = np.mean(frame[valid_mask, :2], axis=0)
                else:
                    anchor_point = np.zeros(2) # 全黑
            # --- 执行中心化 ---
            # 对 x, y 减去锚点
            frame[:, :2] -= anchor_point
            # 将无效点归零
            valid_mask = confs > 0
            frame[~valid_mask, :2] = 0
        return frames.reshape(data.shape[0], -1)

    def _flip_data(self, data):
        data = data.reshape(data.shape[0], 17, 3)
        data[:, :, 0] = 1.0 - data[:, :, 0] 
        for i, j in self.flip_pairs:
            temp = data[:, i, :].copy() 
            data[:, i, :] = data[:, j, :]
            data[:, j, :] = temp
        return data.reshape(data.shape[0], -1)

    def _mask_body_parts(self, data):
        rand = np.random.random()
        temp = data.reshape(data.shape[0], 17, 3)
        if rand < 0.2: 
            temp[:, 11:17, :] = 0.0 # 遮挡下半身
        elif rand < 0.4:
            temp[:, 0:11, :] = 0.0  # 遮挡上半身
        return temp.reshape(data.shape[0], -1)

    def _apply_speed_jitter(self, data, speed_factor):
        seq_len, num_features = data.shape 
        virtual_len = max(1, int(seq_len * speed_factor))
        resampled = cv2.resize(data, (num_features, virtual_len), interpolation=cv2.INTER_LINEAR)
        output = cv2.resize(resampled, (num_features, seq_len), interpolation=cv2.INTER_LINEAR)
        return output

# ==========================================
# 2. 模型定义 (保持不变)
# ==========================================
class ActionModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, bidirectional=True):
        super().__init__()
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                            batch_first=True, dropout=0.5, bidirectional=bidirectional)
        
        fc_input_dim = hidden_size * 2 if bidirectional else hidden_size
        self.fc = nn.Linear(fc_input_dim, num_classes)

    def forward(self, x):
        out, (h_n, c_n) = self.lstm(x)
        if self.bidirectional:
            out = torch.cat((h_n[-2,:,:], h_n[-1,:,:]), dim=1)
        else:
            out = h_n[-1,:,:]
        return self.fc(out)

# ==========================================
# 3. 训练流程 (保持不变)
# ==========================================
def set_seed(seed=42):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

def train_experiment(model_type, full_ds, train_loader, val_loader, device):
    print(f"\n{'='*20} 开始训练模型: {model_type} {'='*20}")
    
    CFG = {
        "lr": 0.0005, "epochs": 100, "patience": 12, 
        "hidden": 128, "layers": 2
    }

    is_bi = True if model_type == "BiLSTM" else False
    model = ActionModel(input_size=51, 
                        hidden_size=CFG["hidden"], 
                        num_layers=CFG["layers"], 
                        num_classes=len(full_ds.classes), 
                        bidirectional=is_bi).to(device)
    
    class_weights = torch.tensor([1.0, 2.0, 1.0, 1.0]).to(device) 
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    optimizer = optim.Adam(model.parameters(), lr=CFG["lr"], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    best_loss = float('inf')
    early_stop_cnt = 0

    for epoch in range(CFG["epochs"]):
        model.train()
        running_train_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item() * inputs.size(0)
        
        model.eval()
        running_val_loss = 0.0
        val_corrects = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                val_corrects += torch.sum(preds == labels.data)

        avg_train_loss = running_train_loss / len(train_loader.dataset)
        avg_val_loss = running_val_loss / len(val_loader.dataset)
        avg_val_acc = val_corrects.double().item() / len(val_loader.dataset)
        
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["val_acc"].append(avg_val_acc)

        print(f"Epoch {epoch+1}: TrainLoss={avg_train_loss:.4f} | ValLoss={avg_val_loss:.4f} | ValAcc={avg_val_acc:.4f}")

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(model.state_dict(), f"best_{model_type}.pth")
            early_stop_cnt = 0
        else:
            early_stop_cnt += 1
        
        scheduler.step()
        if early_stop_cnt >= CFG["patience"]:
            print(f"Early Stopping at epoch {epoch+1}")
            break

    pd.DataFrame(history).to_csv(f"history_{model_type}.csv", index=False)
    
    print(f"\n正在加载 {model_type} 最佳权重进行最终评估...")
    model.load_state_dict(torch.load(f"best_{model_type}.pth"))
    model.eval()
    
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return all_labels, all_preds, full_ds.classes

# ==========================================
# 4. 主程序 (保持不变)
# ==========================================
if __name__ == "__main__":
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if not os.path.exists("dataset_processed"):
        print("错误：找不到 dataset_processed 文件夹！")
        exit()
        
    print("正在加载数据集...")
    full_ds = FitnessDataset("dataset_processed")
    train_size = int(0.8 * len(full_ds))
    val_size = len(full_ds) - train_size
    train_ds, val_ds = random_split(full_ds, [train_size, val_size])
    
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)
    
    results = {}
    for m_type in ["LSTM", "BiLSTM"]:
        labels, preds, class_names = train_experiment(m_type, full_ds, train_loader, val_loader, device)
        results[m_type] = {
            "labels": labels,
            "preds": preds,
            "names": class_names
        }
    
    print("\n\n" + "#"*50)
    print("FINAL COMPARISON REPORT (终端对比)")
    print("#"*50)
    
    for m_type, res in results.items():
        print(f"\n>>> 模型: {m_type}")
        acc = accuracy_score(res["labels"], res["preds"])
        print(f"总体准确率 (Accuracy): {acc:.4f}")
        
        print("\n混淆矩阵 (Confusion Matrix):")
        cm = confusion_matrix(res["labels"], res["preds"])
        print(f"{'':10} " + " ".join([f"{name[:6]:>8}" for name in res["names"]]))
        for i, row in enumerate(cm):
            print(f"{res['names'][i][:10]:>10} " + " ".join([f"{val:>8}" for val in row]))
            
        print("\n分类详情:")
        print(classification_report(res["labels"], res["preds"], target_names=res["names"]))
        print("-" * 50)