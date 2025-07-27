#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import logging
import random
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from nufold.config import model_config
from nufold.model.nufold import Nufold
from nufold.data.data_pipeline import DataPipeline
from nufold.data.feature_pipeline import FeaturePipeline
from nufold.model.openfold.tensor_utils import tensor_tree_map

# 全局logger，在nufold.py中也会用到
logger = logging.getLogger("NufoldDebugger")

def setup_logging(log_file='train.log'):
    """配置日志记录，同时输出到控制台和文件"""
    logger.setLevel(logging.DEBUG)
    
    # 清除已经存在的handlers
    if logger.hasHandlers():
        logger.handlers.clear()

    # 创建文件handler
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(logging.DEBUG)
    
    # 创建控制台handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # 创建formatter并为handlers设置
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # 添加handlers到logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

def set_seed(seed):
    """设置随机种子以保证可复现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def count_parameters(model: torch.nn.Module):
    """计算并返回模型的总参数量和可训练参数量"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

class RNADataset(Dataset):
    """
    自定义RNA数据集
    从文件夹结构中加载和处理用于训练的RNA数据
    """
    def __init__(self, data_dir: str, config):
        super().__init__()
        self.data_dir = data_dir
        self.config = config
        self.sample_dirs = [os.path.join(data_dir, d) for d in os.listdir(data_dir) 
                            if os.path.isdir(os.path.join(data_dir, d))]
        
        # 初始化数据处理 pipeline
        self.data_pipeline = DataPipeline(template_featurizer=None, ss_enabled=True)
        self.feature_processor = FeaturePipeline(config.data)
        
        logger.info(f"找到 {len(self.sample_dirs)} 个训练样本来自目录: {data_dir}")

    def __len__(self):
        return len(self.sample_dirs)

    def __getitem__(self, idx):
        sample_dir = self.sample_dirs[idx]
        sample_name = os.path.basename(sample_dir)
        
        # NEW: 添加详细的调试日志来追踪 __getitem__ 的执行流程
        logger.debug(f"[{sample_name}] 开始处理样本...")
        
        pdb_path = os.path.join(sample_dir, f"{sample_name}.pdb")
        
        if not os.path.exists(pdb_path):
            # 这个错误现在会被明确地记录下来
            logger.error(f"[{sample_name}] PDB文件未找到: {pdb_path}")
            return None
            
        try:
            # --- 步骤 1: Data Pipeline ---
            logger.debug(f"[{sample_name}] 调用 data_pipeline.process_pdb...")
            raw_features = self.data_pipeline.process_pdb(
                pdb_path=pdb_path, 
                alignment_dir=sample_dir, 
                is_distillation=False
            )
            logger.debug(f"[{sample_name}] data_pipeline.process_pdb 完成。")
            
            # NEW: 检查 raw_features 的内容
            if not raw_features:
                logger.error(f"[{sample_name}] data_pipeline 返回了空的 raw_features 字典。")
                return None
            
            logger.debug(f"[{sample_name}] raw_features 键: {list(raw_features.keys())}")
            
            # 检查关键字段是否存在且不为空，这通常是错误的来源
            if 'seq_length' not in raw_features or len(raw_features['seq_length']) == 0:
                logger.error(f"[{sample_name}] 错误的关键点: 'seq_length' 不存在或为空。")
                # 打印相关特征以帮助调试
                for key in ['aatype', 'sequence', 'msa', 'seq_length']:
                    if key in raw_features:
                        logger.debug(f"[{sample_name}] raw_features['{key}'] (shape: {raw_features[key].shape}): {raw_features[key]}")
                    else:
                        logger.debug(f"[{sample_name}] raw_features 中缺少 '{key}'")
                return None

            logger.debug(f"[{sample_name}] raw_features['seq_length'] (shape: {raw_features['seq_length'].shape}): {raw_features['seq_length']}")

            # --- 步骤 2: Feature Pipeline ---
            logger.debug(f"[{sample_name}] 调用 feature_processor.process_features...")
            processed_features = self.feature_processor.process_features(
                raw_features, mode='train'
            )
            logger.debug(f"[{sample_name}] feature_processor.process_features 完成。")
            
            return processed_features
            
        except Exception as e:
            # NEW: 捕获异常时记录完整的追溯信息
            logger.error(f"[{sample_name}] 处理样本时发生异常", exc_info=True)
            return None


def collate_fn(batch):
    """
    自定义collate_fn来过滤掉处理失败的样本
    """
    batch = [item for item in batch if item is not None]
    if not batch:
        return None
    # 由于batch_size=1，我们只需要返回第一个（也是唯一一个）有效的item
    # 并使用 unsqueeze(0) 增加batch维度
    # 注意: 这个collate_fn只适用于batch_size=1的情况
    single_item = batch[0]
    return tensor_tree_map(lambda x: torch.as_tensor(x).unsqueeze(0), single_item)


class LossModule(torch.nn.Module):
    """
    根据NuFold论文和config计算总损失
    """
    def __init__(self, config):
        super().__init__()
        self.config = config.loss

    def forward(self, outputs, batch):
        loss_dict = {}

        # 从batch中只取出最后一次循环的真值(ground truth)
        # 这是为了匹配只有最后一次循环结果的 outputs
        # 我们通过索引 `[..., -1]` 来获取最后一个recycling cycle的数据
        final_batch = tensor_tree_map(lambda t: t[..., -1], batch)

        # 1. FAPE Loss (Frame Aligned Point Error)
        fape_config = self.config.fape
        # 使用 'final_atom_positions' (28原子) 
        pred_positions = outputs['final_atom_positions'].squeeze(0)      # [N, 28, 3]
        # FAPE loss需要比较预测的原子坐标和真实坐标
        # 我们使用一个简化版的FAPE，计算关键原子的坐标偏差
        true_positions = final_batch['all_atom_positions'].squeeze(0) # [N, 28, 3]
        atom_mask = final_batch['all_atom_mask'].squeeze(0) # [N, 28]

        fape_error = torch.sqrt(torch.sum((pred_positions - true_positions)**2, dim=-1) + fape_config.eps)
        clamped_fape_error = torch.clamp(fape_error, max=fape_config.backbone.clamp_distance)
        loss_fape = (torch.sum(clamped_fape_error * atom_mask) / (torch.sum(atom_mask) + 1e-8))
        loss_dict['fape'] = loss_fape * fape_config.weight

        # 2. Masked MSA Loss
        msa_config = self.config.masked_msa
        msa_logits = outputs['masked_msa_logits'] # [B, N_seq, N_res, C_out]
        true_msa = final_batch['true_msa'] # [B, N_seq, N_res]
        bert_mask = final_batch['bert_mask'] # [B, N_seq, N_res]
        
        # 调整形状以适应交叉熵损失
        msa_logits_flat = msa_logits.view(-1, msa_logits.size(-1))
        true_msa_flat = true_msa.view(-1)
        
        # 计算交叉熵损失
        loss_msa = F.cross_entropy(msa_logits_flat, true_msa_flat, reduction='none')
        loss_msa = loss_msa.view_as(true_msa)
        
        # 应用bert_mask
        masked_msa_loss = torch.sum(loss_msa * bert_mask) / (torch.sum(bert_mask) + 1e-8)
        loss_dict['masked_msa'] = masked_msa_loss * msa_config.weight

        # 3. Distogram Loss
        disto_config = self.config.distogram
        disto_logits = outputs['distogram_logits'] # [B, N_res, N_res, C_bins]
        
        # 需要从batch中生成真实的distogram
        # 此处简化：假设我们有 ground truth distogram (实际中需要计算)
        # 鉴于其复杂性，这里我们使用一个占位符
        loss_distogram = torch.tensor(0.0, device=disto_logits.device)
        loss_dict['distogram'] = loss_distogram * disto_config.weight
        
        # ... 可以添加更多损失项，如 lddt, angle_2d 等...
        
        # 计算总损失
        total_loss = sum(loss_dict.values())
        loss_dict['total'] = total_loss
        
        return total_loss, loss_dict


def train(args):
    """主训练函数"""
    setup_logging()
    set_seed(args.seed)

    # 检查并创建检查点目录
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)
        logger.info(f"已创建检查点目录: {args.checkpoint_dir}")

    # 加载模型配置
    # 使用 'initial_training' 预设，并标记为训练模式
    config = model_config('initial_training', train=True)
    config.data.common.max_recycling_iters = args.recycle
    
    # 设置设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")

    # 初始化模型
    model = Nufold(config)
    model = model.to(device)
    logger.info("Nufold模型已成功初始化")

    # NEW: 在此处调用参数统计函数
    total_params, trainable_params = count_parameters(model)
    logger.info(f"模型参数量统计:")
    logger.info(f"  -> 总参数量: {total_params/1e6:.2f}M")
    logger.info(f"  -> 可训练参数量: {trainable_params/1e6:.2f}M")

    # 初始化数据集和数据加载器
    # 注意：由于输入特征的维度可变，batch_size > 1 需要复杂的padding逻辑。
    # 这里我们遵循 AlphaFold 的训练方式，使用等效于 batch_size=1 的方法。
    train_dataset = RNADataset(data_dir=args.train_data_dir, config=config)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=1, # 简化处理，每次加载一个样本
        shuffle=True, 
        num_workers=args.num_workers,
        collate_fn=collate_fn # 使用自定义的collate_fn来处理可能的加载错误
    )

    # 初始化优化器
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # 初始化损失函数模块
    loss_computer = LossModule(config)

    # 开始训练循环
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        
        # 在每个epoch开始时重置显存峰值统计数据
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        # 使用tqdm显示进度条
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", leave=True)
        
        for batch in progress_bar:
            if batch is None:
                logger.warning("一个batch处理失败，已跳过。")
                continue

            # 将数据移动到指定设备
            batch = tensor_tree_map(lambda t: t.to(device), batch)

            # 清零梯度
            optimizer.zero_grad()

            # 前向传播
            outputs = model(batch)

            # 计算损失
            total_loss, loss_dict = loss_computer(outputs, batch)

            # 如果损失有效，则进行反向传播和优化
            if torch.isfinite(total_loss):
                # 反向传播
                total_loss.backward()

                # 梯度裁剪 (防止梯度爆炸)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                # 更新权重
                optimizer.step()

                epoch_loss += total_loss.item()
                
                # 更新进度条的显示信息
                log_info = {k: f"{v.item():.4f}" for k, v in loss_dict.items()}
                progress_bar.set_postfix(log_info)

        # 打印并记录每个epoch的平均损失
        avg_epoch_loss = epoch_loss / len(train_loader) if len(train_loader) > 0 else 0
        logger.info(f"Epoch {epoch+1} 完成, 平均损失: {avg_epoch_loss:.4f}")

        # 记录并打印当前epoch的最大显存占用
        if torch.cuda.is_available():
            # 获取峰值显存占用（单位是字节），转换为GB
            peak_memory_gb = torch.cuda.max_memory_reserved() / (1024**3)
            logger.info(f"Epoch {epoch+1} 峰值显存占用 (Peak Memory Reserved): {peak_memory_gb:.4f} GB")

        # 保存模型检查点
        checkpoint_path = os.path.join(args.checkpoint_dir, f'nufold_epoch_{epoch+1}.pt')
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_epoch_loss,
        }, checkpoint_path)
        logger.info(f"模型检查点已保存至: {checkpoint_path}")

    logger.info("训练完成!")


def main():
    parser = argparse.ArgumentParser(description="训练NuFold模型")
    parser.add_argument('--train_data_dir', type=str, default='./data/train', help='训练数据目录路径')
    parser.add_argument('--checkpoint_dir', type=str, default='./new_checkpoints', help='保存模型检查点的目录')
    parser.add_argument('--lr', type=float, default=1e-3, help='学习率')
    parser.add_argument('--epochs', type=int, default=10, help='训练轮次')
    parser.add_argument('--recycle', type=int, default=3, help='Recycling次数')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--num_workers', type=int, default=0, help='数据加载器的工作线程数(在Windows或macOS上建议设为0)')
    
    args = parser.parse_args()
    
    train(args)

if __name__ == '__main__':
    main()