import torch
import torch.nn as nn
import torch.optim as optim
import time
import math
from tqdm import tqdm

# --- 1. Configurations ---
# Model-specific parameters
MODEL_CONFIG = {
    "image_size": 1024,
    "patch_size": 16,
    "num_classes": 1000,
    "dim": 1024,
    "depth": 6,
    "heads": 8,
    "mlp_dim": 4096,
    "channels": 3,
}

# Training-specific parameters
TRAIN_CONFIG = {
    "batch_size": 8,
    "num_epochs": 3,
    "num_batches_per_epoch": 100,
    "learning_rate": 1e-4,
}

# --- 2.显存监控工具 ---
def print_memory_usage(stage=""):
    if torch.cuda.is_available():
        reserved = torch.cuda.memory_reserved() / 1024**3
        peak_reserved = torch.cuda.max_memory_reserved() / 1024**3
        print(f"[{stage}] "
              f"Current Memory Reserved: {reserved:.2f} GB | "
              f"Peak Memory Reserved: {peak_reserved:.2f} GB")

# --- 3. Simple Vision Transformer (ViT) Model ---
class SimpleViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, channels=3):
        super().__init__()
        image_height, image_width = image_size, image_size
        patch_height, patch_width = patch_size, patch_size
        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        self.to_patch_embedding = nn.Sequential(nn.LayerNorm(patch_dim), nn.Linear(patch_dim, dim), nn.LayerNorm(dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=mlp_dim, batch_first=True, activation='gelu')
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.mlp_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, num_classes))
        self.patch_unfold = nn.Unfold(kernel_size=(patch_height, patch_width), stride=(patch_height, patch_width))

    def forward(self, img):
        x = self.patch_unfold(img).transpose(1, 2)
        x = self.to_patch_embedding(x)
        b, n, _ = x.shape
        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.transformer_encoder(x)
        x = x[:, 0]
        return self.mlp_head(x)


# --- 4. 训练和测试主函数 ---
def main():
    if not torch.cuda.is_available():
        print("错误：未找到 CUDA 设备。此脚本需要 GPU 才能运行。")
        return

    device = torch.device("cuda")
    print(f"使用设备: {torch.cuda.get_device_name(0)}")

    print("正在创建模型...")
    model = SimpleViT(**MODEL_CONFIG).to(device)
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型总参数量: {total_params / 1e6:.2f} M")
    
    print_memory_usage("Model Loaded")

    optimizer = optim.AdamW(model.parameters(), lr=TRAIN_CONFIG["learning_rate"])
    criterion = nn.CrossEntropyLoss()
    
    print("\n--- 开始性能测试 ---")
    start_time = time.time()
    
    for epoch in range(TRAIN_CONFIG["num_epochs"]):
        model.train()
        
        progress_bar = tqdm(
            range(TRAIN_CONFIG["num_batches_per_epoch"]), 
            desc=f"Epoch {epoch+1}/{TRAIN_CONFIG['num_epochs']}",
            unit="batch"
        )

        for i in progress_bar:
            # **FIX**: Create the tensor first, then convert its memory format.
            inputs = torch.randn(
                TRAIN_CONFIG["batch_size"], MODEL_CONFIG["channels"], MODEL_CONFIG["image_size"], MODEL_CONFIG["image_size"],
                device=device
            ).to(memory_format=torch.channels_last)

            labels = torch.randint(0, MODEL_CONFIG["num_classes"], (TRAIN_CONFIG["batch_size"],), device=device)

            optimizer.zero_grad(set_to_none=True)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            vram_reserved_gb = torch.cuda.memory_reserved() / 1024**3
            progress_bar.set_postfix(loss=f"{loss.item():.4f}", vram=f"{vram_reserved_gb:.2f}GB")

    end_time = time.time()
    total_time = end_time - start_time
    total_batches = TRAIN_CONFIG["num_epochs"] * TRAIN_CONFIG["num_batches_per_epoch"]
    avg_batch_time = total_time / total_batches if total_batches > 0 else 0
    
    print("\n--- 测试完成 ---")
    print(f"总耗时: {total_time:.2f} 秒")
    print(f"总批次数: {total_batches}")
    print(f"平均每个Batch耗时: {avg_batch_time * 1000:.2f} 毫秒")
    print_memory_usage("Final")


if __name__ == "__main__":
    main()