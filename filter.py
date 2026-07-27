import os
import random
import warnings
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.models import ResNet18_Weights
from lifelines.utils import concordance_index
import matplotlib
import matplotlib.pyplot as plt
import cv2
from sklearn.model_selection import train_test_split

# =========================
# 0. 全局配置
# =========================
warnings.filterwarnings("ignore", category=UserWarning, message="Glyph .* missing from font")
matplotlib.rcParams["font.family"] = "Times New Roman"
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

# =========================
# 1. 基础配置
# =========================
DATASET_ROOT = r"F:/pyproject/processed_aligned1"
SURVIVAL_CSV = r"F:/pyproject/out2.csv"
SLICE_CSV = r"F:/pyproject/CGGA_172_tuceng.csv"

BATCH_SIZE = 4
EPOCHS = 40
LR = 1e-4
SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

os.makedirs("results", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)

# =========================
# 2. Dataset
# =========================
class DualModalityDataset(Dataset):
    def __init__(self, root_dir, patient_ids, survival_df, slice_df, transform=None):
        self.root_dir = root_dir
        self.patient_ids = patient_ids
        self.survival_df = survival_df.set_index("sample").loc[patient_ids].reset_index()
        slice_df = slice_df.copy()
        slice_df.columns = ["Sample", "Start", "End"]
        self.slice_df = slice_df.set_index("Sample").loc[patient_ids]
        self.transform = transform

    def _load_slices(self, patient_id, modality_folder):
        folder = os.path.join(self.root_dir, patient_id, modality_folder)
        if not os.path.isdir(folder):
            raise FileNotFoundError(f"❌ 目录不存在: {folder}")
        files = sorted(
            [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith((".png", ".jpg"))],
            key=lambda x: int(os.path.basename(x).split("_")[-1].split(".")[0])
        )
        slices = [Image.open(f).convert("RGB") for f in files]
        if self.transform:
            slices = [self.transform(img) for img in slices]
        return torch.stack(slices)

    def __getitem__(self, idx):
        pid = self.patient_ids[idx]
        surv_row = self.survival_df.iloc[idx]
        slice_row = self.slice_df.loc[pid]
        start, end = int(slice_row["Start"]), int(slice_row["End"])
        ce = self._load_slices(pid, "CE_slices")[start:end + 1]
        t2 = self._load_slices(pid, "T2_slices")[start:end + 1]
        return {
            "ce": ce, "t2": t2,
            "time": torch.tensor(surv_row["risk_raw"], dtype=torch.float),
            "event": torch.tensor(surv_row["risk_01"], dtype=torch.float),
            "pid": pid
        }

    def __len__(self):
        return len(self.patient_ids)

# =========================
# 3. Collate
# =========================
def dynamic_collate(batch):
    return {
        "ce": [b["ce"] for b in batch],
        "t2": [b["t2"] for b in batch],
        "time": torch.stack([b["time"] for b in batch]),
        "event": torch.stack([b["event"] for b in batch]),
        "pid": [b["pid"] for b in batch]
    }

# =========================
# 4. 模型
# =========================
class SingleBranchFeatureModel(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        return self.pool(self.backbone(x)).squeeze(-1).squeeze(-1)

class SliceAttentionPooling(nn.Module):
    def __init__(self, feat_dim=512):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(feat_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return torch.sum(torch.softmax(self.attn(x), dim=0) * x, dim=0)

class SurvivalRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.ce_model = SingleBranchFeatureModel()
        self.t2_model = SingleBranchFeatureModel()
        self.ce_pool = SliceAttentionPooling()
        self.t2_pool = SliceAttentionPooling()
        self.head = nn.Sequential(
            nn.Linear(1024, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1)
        )

    def forward(self, ce_list, t2_list):
        preds = []
        for ce, t2 in zip(ce_list, t2_list):
            fused = torch.cat([
                self.ce_pool(self.ce_model(ce)),
                self.t2_pool(self.t2_model(t2))
            ], dim=0)
            preds.append(self.head(fused))
        return torch.relu(torch.stack(preds).squeeze(-1)) + 0.1

# =========================
# 5. 损失函数
# =========================
def survival_regression_loss(pred, time, event):
    mask = event.bool()
    if mask.sum() == 0:
        return torch.tensor(0.0, device=pred.device, requires_grad=True)
    return nn.L1Loss()(pred[mask], time[mask])

# =========================
# 6. C-index
# =========================
def compute_c_index(model, dataloader, device):
    model.eval()
    risks, times, events = [], [], []
    with torch.no_grad():
        for b in dataloader:
            preds = model(
                [x.to(device) for x in b["ce"]],
                [x.to(device) for x in b["t2"]]
            ).cpu().numpy()
            risks.extend(preds)
            times.extend(b["time"].cpu().numpy())
            events.extend(b["event"].cpu().numpy())
    return concordance_index(times, -np.array(risks), events)

# =========================
# 7. 数据加载
# =========================
survival_df = pd.read_csv(SURVIVAL_CSV)
slice_df = pd.read_csv(SLICE_CSV, encoding="gb18030")
patients = survival_df["sample"].tolist()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

train_patients, temp = train_test_split(patients, test_size=0.2, random_state=SEED)
val_patients, test_patients = train_test_split(temp, test_size=0.5, random_state=SEED)

train_ds = DualModalityDataset(DATASET_ROOT, train_patients, survival_df, slice_df, transform)
val_ds = DualModalityDataset(DATASET_ROOT, val_patients, survival_df, slice_df, transform)
test_ds = DualModalityDataset(DATASET_ROOT, test_patients, survival_df, slice_df, transform)

train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=dynamic_collate)
val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=dynamic_collate)
test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=dynamic_collate)

# =========================
# 8. 训练（含模型保存）
# =========================
model = SurvivalRegressionModel().to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LR)

best_val_loss = float("inf")

for epoch in range(EPOCHS):
    # ---- Train ----
    model.train()
    train_loss = 0
    for b in train_dl:
        pred = model([x.to(DEVICE) for x in b["ce"]],
                     [x.to(DEVICE) for x in b["t2"]])
        loss = survival_regression_loss(
            pred, b["time"].to(DEVICE), b["event"].to(DEVICE)
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    # ---- Validation ----
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for b in val_dl:
            pred = model([x.to(DEVICE) for x in b["ce"]],
                         [x.to(DEVICE) for x in b["t2"]])
            val_loss += survival_regression_loss(
                pred, b["time"].to(DEVICE), b["event"].to(DEVICE)
            ).item()
    val_loss /= len(val_dl)

    c_idx = compute_c_index(model, val_dl, DEVICE)

    print(f"Epoch {epoch+1}/{EPOCHS} | "
          f"Train Loss: {train_loss/len(train_dl):.4f} | "
          f"Val Loss: {val_loss:.4f} | "
          f"C-index: {c_idx:.4f}")

    # ---- Save Best Model ----
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_loss,
        }, "checkpoints/best_model.pth")
        print("✅ 保存最优模型")

print("✅ 训练完成，最优模型已保存至 checkpoints/best_model.pth")

# =========================
# 9. 加载最优模型（关键）
# =========================
checkpoint = torch.load("checkpoints/best_model.pth", map_location=DEVICE)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
print(f"✅ 已加载最优模型（Epoch {checkpoint['epoch']+1}, Val Loss: {checkpoint['val_loss']:.4f}）")

# =========================
# 10. 2D Grad‑CAM
# =========================
def generate_gradcam(submodel, x):
    x = x.clone().detach().requires_grad_(True)
    feat = submodel.backbone(x)
    score = feat.sum()
    submodel.zero_grad()
    score.backward()
    grads = x.grad.abs().mean(dim=1).squeeze(0)
    cam = (grads - grads.min()) / (grads.max() - grads.min() + 1e-8)
    return cam.cpu()

def save_gradcam_figure(img_tensor, cam, save_path):
    img = img_tensor.permute(1, 2, 0).cpu().numpy()
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)

    cam_v = cam.detach().cpu().numpy()
    cam_v = (cam_v - cam_v.min()) / (cam_v.max() - cam_v.min() + 1e-8)

    cam_color = cv2.applyColorMap(np.uint8(255 * cam_v), cv2.COLORMAP_JET)
    cam_color = cv2.cvtColor(cam_color, cv2.COLOR_BGR2RGB)
    cam_color = cv2.resize(cam_color, (img.shape[1], img.shape[0]))
    overlay = img * 0.5 + cam_color / 255.0 * 0.5

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(img)
    axes[0].set_title("Original", fontsize=11, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(overlay)
    axes[1].set_title("Overlay", fontsize=11, fontweight="bold")
    axes[1].axis("off")

    ax_inset = axes[1].inset_axes([0.72, 0.72, 0.25, 0.25])
    grad = np.linspace(0, 1, 256).reshape(-1, 1)
    ax_inset.imshow(grad, aspect="auto", cmap="jet", extent=[0, 1, 0, 1])
    ax_inset.set_xticks([])
    ax_inset.set_yticks([0, 0.5, 1])
    ax_inset.set_yticklabels(["Low", "Med", "High"], fontsize=6, fontweight="bold")
    ax_inset.tick_params(axis="y", length=0, pad=1)
    ax_inset.set_title("Risk", fontsize=7, fontweight="bold", pad=2)

    plt.suptitle("Grad‑CAM Visualization", fontsize=12, fontweight="bold", y=1.0)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path, format="pdf", bbox_inches="tight")
    plt.close()

# =========================
# 11. 3D Grad‑CAM（纯 Overlay 堆叠）
# =========================
def save_3d_gradcam_stacked(cam, save_path, num_slices=7, z_spacing=100):
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    h, w = cam.shape
    X, Y = np.meshgrid(np.arange(w), np.arange(h))

    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection="3d")

    for i in range(num_slices):
        z_level = i * z_spacing
        Z_layer = np.full_like(cam, z_level)
        ax.plot_surface(
            X, Y, Z_layer,
            cmap="jet",
            facecolors=plt.cm.jet(cam),
            shade=False,
            alpha=0.7,
            linewidth=0,
            antialiased=True,
            zorder=i
        )

    ax.view_init(elev=25, azim=-60)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.set_zlim(0, num_slices * z_spacing + 0.1)
    ax.set_box_aspect([w, h, num_slices * z_spacing])

    mappable = plt.cm.ScalarMappable(cmap="jet")
    mappable.set_array(cam)
    cbar = fig.colorbar(mappable, ax=ax, shrink=0.5, pad=0.1)
    cbar.set_label("Activation Intensity", fontsize=10)
    cbar.set_ticks([0, 0.5, 1])
    cbar.set_ticklabels(["Low", "Med", "High"], fontsize=8)

    ax.set_title("3D Stacked Grad‑CAM Overlays", fontsize=12, fontweight="bold", y=0.95)
    plt.tight_layout()
    plt.savefig(save_path, format="pdf", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ 3D Grad‑CAM 已保存: {save_path}")

# =========================
# 12. 生成结果（仅目标病人）
# =========================
TARGET_PATIENTS = {"CGGA_1198", "CGGA_P179"}
os.makedirs("results/gradcam_3d", exist_ok=True)

for dl, tag in [(train_dl, "train"), (val_dl, "val"), (test_dl, "test")]:
    for b in dl:
        for i in range(len(b["pid"])):
            pid = b["pid"][i]
            if pid not in TARGET_PATIENTS:
                continue

            ce_img = b["ce"][i][0].cpu()
            t2_img = b["t2"][i][0].cpu()

            ce_cam = generate_gradcam(model.ce_model, ce_img.unsqueeze(0).to(DEVICE))
            t2_cam = generate_gradcam(model.t2_model, t2_img.unsqueeze(0).to(DEVICE))

            save_gradcam_figure(ce_img, ce_cam, f"results/gradcam_3d/{pid}_CE_2D.pdf")
            save_gradcam_figure(t2_img, t2_cam, f"results/gradcam_3d/{pid}_T2_2D.pdf")

            save_3d_gradcam_stacked(
                ce_cam.numpy(),
                f"results/gradcam_3d/{pid}_CE_3D.pdf",
                num_slices=7,
                z_spacing=100
            )
            save_3d_gradcam_stacked(
                t2_cam.numpy(),
                f"results/gradcam_3d/{pid}_T2_3D.pdf",
                num_slices=7,
                z_spacing=100
            )

            print(f"✅ Grad‑CAM 完成: {pid}")

print("✅ 全部完成")