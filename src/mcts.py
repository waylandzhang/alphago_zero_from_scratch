import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass

from tqdm import tqdm

from src import govars
from src import environment

class ResBlock(nn.Module):
    """残差块实现"""
    def __init__(self, num_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        return F.relu(x)

class GoNeuralNetwork(nn.Module):
    def __init__(self, board_size: int = 9, num_channels: int = 128, num_res_blocks: int = 10):
        super().__init__()
        self.board_size = board_size
        """
        输入层：6个特征平面:
        0 - Black pieces
        1 - White pieces
        2 - Turn (0 - black, 1 - white) 
        3 - Invalid moves
        4 - Previous move was pass
        5 - Game over
        """
        def _initialize_weights(m):
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)
        self.apply(_initialize_weights)

        self.conv_input = nn.Sequential(
            nn.Conv2d(govars.NUM_CHNLS, num_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_channels),
            nn.ReLU()
        )

        # 残差层
        self.res_blocks = nn.ModuleList([
            ResBlock(num_channels) for _ in range(num_res_blocks)
        ])

        # 策略头
        self.policy_head = nn.Sequential(
            nn.Conv2d(num_channels, 2, kernel_size=1),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2 * board_size * board_size, board_size * board_size + 1)  # +1 for pass move
        )

        # 价值头
        self.value_head = nn.Sequential(
            nn.Conv2d(num_channels, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(board_size * board_size, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Tanh()
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.conv_input(x)
        for res_block in self.res_blocks:
            x = res_block(x)

        policy = self.policy_head(x)

        # 添加一些启发式规则有助于在小棋盘上的训练前期稳定些
        batch_size = x.shape[0]
        board_size = self.board_size
        # 1. 天元和星位点有较高优先级
        star_points = torch.zeros((batch_size, board_size * board_size + 1), device=x.device)
        if board_size == 9:
            stars = [20, 24, 28, 40, 44, 48, 60, 64, 68]  # 9x9棋盘的星位点
            star_points[:, stars] = 1.0
        # 2. 边角位置优先级降低
        edge_penalty = torch.ones((batch_size, board_size * board_size + 1), device=x.device)
        for i in range(board_size):
            for j in range(board_size):
                if i == 0 or i == board_size-1 or j == 0 or j == board_size-1:
                    idx = i * board_size + j
                    edge_penalty[:, idx] = 0.8
        policy = policy * star_points * edge_penalty

        # 3. 输出的策略函数为 总和100%的 概率分布
        policy = F.softmax(policy, dim=1)
        # 4. 输出的价值函数为 [-1, 1] 之间的值
        value = self.value_head(x)

        return policy, value

@dataclass
class MCTSNode:
    """蒙特卡洛搜索树节点"""
    prior: float
    state: Optional[np.ndarray] = None
    parent: Optional['MCTSNode'] = None
    children: Dict[Tuple[int, int], 'MCTSNode'] = None
    visit_count: int = 0
    value_sum: float = 0.0
    is_expanded: bool = False

    def __post_init__(self):
        if self.children is None:
            self.children = {}

    def expand(self, state: np.ndarray, policy: np.ndarray):
        """扩展节点"""
        self.is_expanded = True
        self.state = state

        # 为每个可能的动作创建子节点
        policy_size = len(policy)
        board_size = int(np.sqrt(policy_size - 1))

        for action in range(policy_size):
            if action == policy_size - 1:  # Pass move 弃权
                child_action = (-1, -1)
            else:
                child_action = (action // board_size, action % board_size)
            self.children[child_action] = MCTSNode(
                prior=policy[action],
                parent=self
            )

    @property
    def value(self) -> float:
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

    def select_child(self, c_puct: float) -> Tuple[Tuple[int, int], 'MCTSNode']:
        """选择最佳子节点"""
        best_score = float('-inf')
        best_action = None
        best_child = None

        for action, child in self.children.items():
            # UCB 置信度上届 = Q(s, a) + c_puct * P(s, a) * sqrt(N(s)) / (1 + N(s, a)) 公式
            score = child.value + c_puct * child.prior * np.sqrt(self.visit_count) / (1 + child.visit_count)

            if score > best_score:
                best_score = score
                best_action = action
                best_child = child

        return best_action, best_child

class MCTS:
    def __init__(self, model, board_size, num_simulations=800, c_puct=2.0, device='cpu'):
        self.model = model
        self.board_size = board_size
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.device = device

    def run(self, state: np.ndarray) -> np.ndarray:
        """执行MCTS搜索"""
        root = MCTSNode(prior=0)  # 创建一个根节点

        # 评估根节点
        """
        state_tensor 是一个形状为 (1, 6, BOARD_SIZE, BOARD_SIZE)的包含棋盘状态的张量
        1 表示批次大小（batch size），因为在评估时通常只处理一个棋盘状态。
        6 表示特征平面数量，对应上述的六种信息：
            - 黑棋的位置
            - 白棋的位置
            - 当前轮到哪一方下棋（黑或白）
            - 非法动作的位置
            - 上一步是否是弃权
            - 游戏是否结束
        BOARD_SIZE 表示棋盘的边长（例如 9 表示 9x9 的棋盘）。
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            policy, _ = self.model(state_tensor)
            policy = policy[0].cpu().numpy()

        # 将策略与合法动作相乘
        valid_moves = environment.valid_moves(state)
        policy = policy * valid_moves  # 过滤非法动作

        # 重新归一化
        policy_sum = np.sum(policy)
        if policy_sum > 0:
            policy = policy / policy_sum
        else:
            # 如果没有合法动作，则全部概率赋给pass动作
            policy = np.zeros_like(policy)
            policy[-1] = 1.0

        root.expand(state, policy)

        # 执行蒙特卡洛树搜索
        pbar = tqdm(range(self.num_simulations),
                    desc="MCTS simulations",
                    leave=False,
                    position=2)

        for _ in pbar:
            node = root
            search_path = [node]
            current_state = state.copy()

            # Selection - 选择阶段
            """
            从根节点开始，沿着树向下选择子节点，直到到达一个未扩展的节点或游戏结束。
            """
            while node.is_expanded and not environment.game_ended(current_state):
                action, node = node.select_child(self.c_puct)

                # 转换action到1D索引
                if action == (-1, -1):  # Pass move 弃权棋
                    action_idx = self.board_size * self.board_size
                else:
                    action_idx = action[0] * self.board_size + action[1]

                # 验证移动的合法性
                valid_moves = environment.valid_moves(current_state)
                if valid_moves[action_idx] == 0:
                    # 如果选择了非法移动，强制选择pass
                    action_idx = self.board_size * self.board_size

                current_state = environment.next_state(current_state, action_idx)
                search_path.append(node)

            # Expansion and Evaluation - 扩展和评估阶段
            """
            如果到达的节点未扩展且游戏未结束，则扩展该节点并评估其状态。评估结果包括策略函数和价值函数
            """
            value = 0
            if environment.game_ended(current_state):
                value = environment.winning(current_state)
            else:
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(current_state).unsqueeze(0).to(self.device)
                    new_policy, value = self.model(state_tensor)
                    new_policy = new_policy[0].cpu().numpy()
                    value = value[0].item()

                    # 过滤非法动作
                    valid_moves = environment.valid_moves(current_state)
                    new_policy = new_policy * valid_moves
                    policy_sum = np.sum(new_policy)
                    if policy_sum > 0:
                        new_policy = new_policy / policy_sum
                    else:
                        new_policy = np.zeros_like(new_policy)
                        new_policy[-1] = 1.0

                    node.expand(current_state, new_policy)  # 执行MCTSNode.expand函数，继续扩展子节点

            # Backpropagation - 反向传播阶段
            """
            将评估结果（价值）沿着搜索路径向上传播，更新每个节点的访问次数和价值总和。
            """
            for node in reversed(search_path):
                node.value_sum += value
                node.visit_count += 1
                value = -value

        # 计算最终的动作概率分布
        action_probs = np.zeros(self.board_size * self.board_size + 1)

        # 仅考虑合法动作
        valid_moves = environment.valid_moves(state)

        for action, child in root.children.items():
            if action == (-1, -1):  # Pass move 弃权
                action_probs[-1] = child.visit_count
            else:
                idx = action[0] * self.board_size + action[1]
                if valid_moves[idx]:  # 只统计合法动作的访问次数
                    action_probs[idx] = child.visit_count

        # 重新归一化
        visit_sum = np.sum(action_probs)
        if visit_sum > 0:
            action_probs = action_probs / visit_sum
        else:
            action_probs[-1] = 1.0  # 如果没有合法动作，则pass

        # 更新进度条信息
        pbar.set_postfix({
            'value': f'{root.value:.3f}',
            'visits': root.visit_count
        })

        """
        返回一个包含每个可能动作的概率分布的数组。（形状为 (BOARD_SIZE * BOARD_SIZE + 1)）
        它表示在当前状态下，蒙特卡洛树搜索（MCTS）算法推荐的每个动作的选择概率。
        """
        return action_probs


