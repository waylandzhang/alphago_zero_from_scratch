# AlphaGo Zero 教学实现

基于 DeepMind 论文《Mastering the Game of Go without Human Knowledge》的 PyTorch 实现版本，实现了一个不需要人类知识的围棋 AI 系统。

## 🌟 主要特性

- 纯自我对弈的强化学习系统
- 完整的蒙特卡洛树搜索（MCTS）实现
- 双头神经网络架构（策略网络和价值网络）
- 完整的[围棋规则](https://github.com/huangeddie/GymGo/blob/master/gym_go/gogame.py)
- 包含自对弈、训练和评估的完整训练流程
- 支持模型检查点保存和训练恢复

## 🏗️ 系统架构

### 神经网络结构

```
神经网络架构
├── 共享层（特征提取）
│   ├── 输入通道：当前方棋子位置、对手棋子位置、合法落子位置等6个通道
│   ├── 初始卷积层
│   └── 残差块
├── 策略头
│   ├── 输入：提取的特征
│   ├── 输出：落子概率分布
│   └── 损失函数：与MCTS策略的交叉熵
└── 价值头
    ├── 输入：提取的特征
    ├── 输出：局面评估 [-1, 1]
    └── 损失函数：均方误差
```

### MCTS 实现细节

- 基于 UCB 公式的节点选择
- 神经网络引导的树扩展
- 价值回传机制
- 动态温度参数调整
- 处理 pass 动作

## 💻 实现细节

### 训练流程

每个训练周期包含三个阶段：

1. **自对弈阶段**
   - 每个周期对弈局数：20
   - 棋盘大小：9x9
   - 每局最大步数：200
   - 每步 MCTS 模拟次数：400

2. **训练阶段**
   ```python
   训练参数：
   - 批次大小：64
   - 学习率：0.001
   - 权重衰减：1e-4
   - 经验回放池大小：10,000
   ```

3. **评估阶段**（每5个周期进行一次）
   - 当前模型对战先前版本
   - 评估局数：10
   - 跟踪胜率变化

### 数据结构

每局游戏生成的训练数据格式：
```python
{
    'state': ndarray,     # 棋盘状态 (通道数 × 棋盘大小 × 棋盘大小)
    'policy': ndarray,    # 落子概率分布 (棋盘大小 * 棋盘大小 + 1)
    'value': float        # 对局结果 (-1 或 1)
}
```

## 🚀 快速开始

### 环境要求

```bash
pip install torch numpy tqdm
```

### 运行训练

```bash
python train.py
```

### 配置参数

可在 `train.py` 中修改的主要参数：
```python
参数配置 = {
    'board_size': 9,            # 棋盘大小
    'num_channels': 128,        # 通道数
    'num_res_blocks': 6,        # 残差块数量
    'num_simulations': 400,     # 模拟次数
    'replay_buffer_size': 10000,# 回放池大小
    'batch_size': 64,          # 批次大小
    'num_epochs': 100,         # 训练周期
    'games_per_epoch': 20      # 每周期游戏数
}
```

## 📊 训练指标

训练过程会跟踪以下指标：
- 策略损失
- 价值损失
- 总体损失
- 对抗先前版本的胜率
- 平均对局长度
- 自对弈游戏统计数据

## 🔍 核心组件

### 围棋环境 (`environment.py`)
- 完整的围棋规则实现
- 提子规则检测
- 劫争处理
- 领地计算

### MCTS (`mcts.py`)
- 树节点管理
- 基于 UCB 的选择
- 神经网络引导的模拟
- 价值回传机制

### 神经网络 (`mcts.py`)
- 残差网络架构
- 策略和价值双头设计
- 输入特征面处理

## 📈 性能优化

- 神经网络评估的批处理
- 高效的 MCTS 实现
- GPU 加速
- 并行自对弈（计划中）

## 🛣️ 未来计划

- [ ] 实现并行自对弈
- [ ] 添加与训练模型的对弈HTML互动界面
- [ ] 改进领地计算算法

## 📝 说明

本实现专注于通过纯自我对弈学习围棋，遵循 AlphaGo Zero 的核心原则。考虑到普通硬件的计算能力限制，当前版本使用了相比原论文更小的网络结构和更少的模拟次数。主要为教学目的。

## 📖 参考资料

- [Mastering the Game of Go without Human Knowledge](https://www.nature.com/articles/nature24270)
- [AlphaGo Zero: Starting from Scratch](https://deepmind.com/blog/article/alphago-zero-starting-scratch)
- [GymGo](https://github.com/huangeddie/GymGo)