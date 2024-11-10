import torch
import numpy as np
from typing import List, Tuple, Dict, Optional
import os
import time
from torch.utils.data import Dataset, DataLoader
import random
from collections import deque
import json
from datetime import datetime
from tqdm import tqdm
from src.mcts import MCTS, GoNeuralNetwork
from src import environment
from src import govars

class GameWrapper:
    def __init__(self, board_size):
        self.board_size = board_size
        self.board = None
        self.consecutive_passes = 0
        self.moves = 0
        self.max_moves = board_size * board_size * 2  # 设置最大步数
        self.reset()

    def reset(self):
        self.board = environment.init_state(self.board_size)
        self.consecutive_passes = 0
        self.moves = 0
        return self.board

    def step(self, action):
        self.board = environment.next_state(self.board, action)
        self.moves += 1

        # 更新连续pass计数
        if action == self.board_size * self.board_size:  # Pass动作
            self.consecutive_passes += 1
        else:
            self.consecutive_passes = 0

        return self.board

    def get_valid_moves(self):
        return environment.valid_moves(self.board)

    def is_game_over(self):
        # 必要的终局条件
        if self.consecutive_passes >= 2:
            return True

        if self.moves >= self.max_moves:
            return True

        valid_moves = self.get_valid_moves()
        if np.sum(valid_moves) == 1 and valid_moves[-1] == 1:
            return True

        return environment.game_ended(self.board)

    def get_winner(self):
        black_area, white_area = environment.areas(self.board)
        diff = black_area - white_area - 6.5  # 6.5是贴目
        return 1 if diff > 0 else -1

class SelfPlayGame:
    def __init__(self, mcts: MCTS):
        self.mcts = mcts
        self.board_size = mcts.board_size
        self.game = GameWrapper(self.board_size)
        self.training_data = []

    def play_game(self, game_idx: Optional[int] = None) -> List[Dict]:
        state = self.game.reset()
        game_history = []
        move_count = 0

        # 创建游戏步骤的进度条
        game_desc = f"Game {game_idx}" if game_idx is not None else "Game moves"
        move_pbar = tqdm(desc=game_desc,
                         leave=False,
                         position=0)

        while not self.game.is_game_over():
            valid_moves = self.game.get_valid_moves()
            policy = self.mcts.run(state)

            # print(f"\nBefore move {move_count}:")
            # print(f"Game state:\n{environment.str(state)}")
            # print(f"Is game over: {self.game.is_game_over()}")

            # 打印更多调试信息
            black_area, white_area = environment.areas(state)
            move_pbar.set_postfix({
                'moves': move_count,
                'B/W': f"{black_area}/{white_area}",
                'valid': f"{np.sum(valid_moves)}",
                'passes': self.game.consecutive_passes
            })

            # 确保策略只包含合法动作
            policy = policy * valid_moves
            if np.sum(policy) > 0:
                policy = policy / np.sum(policy)
            else:
                # 如果没有合法动作，强制pass
                policy = np.zeros_like(policy)
                policy[-1] = 1.0

            # 记录当前状态
            game_history.append({
                'state': state.copy(),
                'policy': policy,
                'current_player': environment.turn(state)
            })

            # 选择动作
            temperature = 1.0 if move_count < 30 else 0.1
            if temperature == 1.0:
                # 添加 Dirichlet 噪声增加探索
                policy = 0.75 * policy + 0.25 * np.random.dirichlet([0.3] * len(policy))
                policy = policy * valid_moves  # 确保只有合法动作
                policy = policy / np.sum(policy)
                action = np.random.choice(len(policy), p=policy)
            else:
                legal_policy = policy * valid_moves
                action = np.argmax(legal_policy)

            # 执行动作
            try:
                old_state = state.copy()  # 保存旧状态用于对比
                state = self.game.step(action)
                move_count += 1
                move_pbar.update(1)

                # 输出棋盘状态调试打印，正式训练可以注释掉
                print(f"After move {move_count}:")
                print(f"Game state:\n{environment.str(state)}")
                print(f"Is game over: {self.game.is_game_over()}")
                print(f"Consecutive passes: {self.game.consecutive_passes}")


            except AssertionError as e:
                print(f"Warning: Invalid move attempted {action}, choosing random valid move")
                valid_actions = np.where(valid_moves == 1)[0]
                if len(valid_actions) > 0:
                    action = np.random.choice(valid_actions)
                    state = self.game.step(action)
                else:
                    # 如果没有合法动作，执行pass
                    pass_action = self.board_size * self.board_size
                    state = self.game.step(pass_action)

        # 获取游戏结果
        game_outcome = self.game.get_winner()

        # 记录训练数据
        for historical_state in game_history:
            player = historical_state['current_player']
            value = game_outcome if player == 0 else -game_outcome
            self.training_data.append({
                'state': historical_state['state'],
                'policy': historical_state['policy'],
                'value': value
            })

        move_pbar.close()
        return self.training_data

class ReplayBuffer:
    def __init__(self, capacity: int, device: str, board_size: int):
        self.device = device
        self.board_size = board_size
        # 修改state维度以匹配新的环境
        self.states = np.zeros((capacity, govars.NUM_CHNLS, board_size, board_size), dtype=np.float32)
        self.policies = np.zeros((capacity, board_size * board_size + 1), dtype=np.float32)
        self.values = np.zeros(capacity, dtype=np.float32)
        self.capacity = capacity
        self.size = 0
        self.position = 0

    def add(self, data: List[Dict]):
        for item in data:
            if self.size < self.capacity:
                self.size += 1

            self.states[self.position] = item['state']
            self.policies[self.position] = item['policy']
            self.values[self.position] = item['value']

            self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        indices = np.random.randint(0, self.size, size=batch_size)

        states = torch.from_numpy(self.states[indices]).to(self.device)
        policies = torch.from_numpy(self.policies[indices]).to(self.device)
        values = torch.from_numpy(self.values[indices]).to(self.device)

        return states, policies, values

class AlphaGoZeroTrainer:
    def __init__(self,
                board_size: int = 9,
                num_channels: int = 128,
                num_res_blocks: int = 10,
                num_simulations: int = 200,
                replay_buffer_size: int = 500000,
                batch_size: int = 128,
                num_epochs: int = 100,
                games_per_epoch: int = 25,
                eval_games: int = 10,
                checkpoint_dir: str = 'checkpoints',
                learning_rate: float = 0.001,
                weight_decay: float = 1e-4):

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.board_size = board_size
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.games_per_epoch = games_per_epoch
        self.eval_games = eval_games
        self.checkpoint_dir = checkpoint_dir

        # Initialize models
        self.model = GoNeuralNetwork(board_size, num_channels, num_res_blocks).to(self.device)
        self.eval_model = GoNeuralNetwork(board_size, num_channels, num_res_blocks).to(self.device)

        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        # 初始化MCTS
        game_wrapper = GameWrapper(board_size)
        self.mcts = MCTS(
            model=self.model,
            board_size=self.board_size,
            num_simulations=num_simulations,
            device=self.device
        )

        # Initialize replay buffer
        self.replay_buffer = ReplayBuffer(
            replay_buffer_size,
            self.device,
            board_size
        )

        # Create checkpoint directory
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Training history
        self.training_history = {
            'policy_loss': [],
            'value_loss': [],
            'total_loss': [],
            'eval_win_rate': []
        }

    def generate_self_play_data(self) -> int:
        """Generate self-play data and return number of moves"""
        self.model.eval()
        self_play = SelfPlayGame(self.mcts)
        training_data = self_play.play_game()
        self.replay_buffer.add(training_data)

        return len(training_data)

    def train_epoch(self):
        """Train one epoch"""
        if self.replay_buffer.size < self.batch_size:
            return 0, 0, 0

        self.model.train()
        num_batches = min(self.replay_buffer.size // self.batch_size, 1000)

        total_loss = 0
        total_policy_loss = 0
        total_value_loss = 0

        pbar = tqdm(range(num_batches), desc="Training batches",position=1, leave=False)

        for _ in pbar:
            states, policies, values = self.replay_buffer.sample(self.batch_size)

            # Forward pass
            self.optimizer.zero_grad()
            pred_policies, pred_values = self.model(states)

            # Calculate losses
            policy_loss = -torch.mean(torch.sum(policies * torch.log(pred_policies + 1e-8), dim=1))
            value_loss = torch.mean((values - pred_values.squeeze()) ** 2)
            loss = policy_loss + value_loss

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            # Update statistics
            total_loss += loss.item()
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()

            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'policy_loss': f'{policy_loss.item():.4f}',
                'value_loss': f'{value_loss.item():.4f}'
            })

        return (total_loss / num_batches,
                total_policy_loss / num_batches,
                total_value_loss / num_batches)

    def evaluate(self) -> float:
        """Evaluate current model against evaluation model"""
        self.model.eval()
        self.eval_model.eval()

        wins = 0
        draws = 0

        for _ in tqdm(range(self.eval_games), desc="Evaluating"):
            game = GameWrapper(self.board_size)
            current_mcts = MCTS(
                model=self.model,
                board_size=self.board_size,
                num_simulations=self.mcts.num_simulations,
                device=self.device
            )

            eval_mcts = MCTS(
                model=self.eval_model,
                board_size=self.board_size,
                num_simulations=self.mcts.num_simulations,
                device=self.device
            )

            state = game.reset()

            while not game.is_game_over():
                # 选择当前玩家的MCTS
                current_player = environment.turn(state)
                mcts = current_mcts if current_player == 0 else eval_mcts

                # 获取策略并选择动作
                policy = mcts.run(state)
                action = np.argmax(policy)

                # 执行动作
                state = game.step(action)

            # 判断胜负
            game_outcome = game.get_winner()
            if game_outcome == 1:  # Current model (Black) wins
                wins += 1
            elif game_outcome == 0:  # Draw
                draws += 0.5

        return (wins + draws) / self.eval_games

    def save_checkpoint(self, epoch: int, metrics: Dict):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics
        }

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        path = os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch}_{timestamp}.pt')
        torch.save(checkpoint, path)

        # Save training history
        history_path = os.path.join(self.checkpoint_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f)

    def train(self):
        """Main training loop"""
        print(f"\n=== Training Configuration ===")
        print(f"Board size: {self.board_size}x{self.board_size}")
        print(f"MCTS simulations: {self.mcts.num_simulations}")
        print(f"Games per epoch: {self.games_per_epoch}")
        print(f"Number of epochs: {self.num_epochs}")
        print(f"Device: {self.device}")
        print(f"Model channels: {self.model.conv_input[0].out_channels}")
        print(f"Number of residual blocks: {len(self.model.res_blocks)}")
        print("=" * 50)

        total_start_time = time.time()

        try:
            epoch_pbar = tqdm(range(self.num_epochs), desc="Training epochs", position=2, leave=True)
            for epoch in epoch_pbar:
                epoch_start_time = time.time()
                print(f"\nEpoch {epoch + 1}/{self.num_epochs}")
                # Self-play phase
                print("\nSelf-play phase:")
                total_moves = 0
                game_pbar = tqdm(range(self.games_per_epoch),
                               desc="Self-play games",
                               leave=False,
                               position=1)
                for game_idx in game_pbar:
                    moves = self.generate_self_play_data()
                    total_moves += moves
                    # 更新游戏进度条信息
                    game_pbar.set_postfix({
                        'moves': moves,
                        'avg_moves': f"{total_moves/(game_idx+1):.1f}"
                    })

                game_pbar.close()
                # Training phase
                print("\nTraining phase:")
                total_loss, policy_loss, value_loss = self.train_epoch()

                # Update training history
                self.training_history['policy_loss'].append(policy_loss)
                self.training_history['value_loss'].append(value_loss)
                self.training_history['total_loss'].append(total_loss)

                # Evaluation phase
                if (epoch + 1) % 5 == 0:
                    print("\nEvaluation phase:")
                    self.eval_model.load_state_dict(self.model.state_dict())
                    win_rate = self.evaluate()
                    self.training_history['eval_win_rate'].append(win_rate)
                    print(f"Win rate against previous version: {win_rate:.3f}")

                # Save checkpoint
                self.save_checkpoint(epoch + 1, {
                    'total_loss': total_loss,
                    'policy_loss': policy_loss,
                    'value_loss': value_loss,
                    'replay_buffer_size': self.replay_buffer.size
                })

                epoch_time = time.time() - epoch_start_time
                print(f"\nEpoch completed in {epoch_time:.1f}s")

        except KeyboardInterrupt:
            print("\nTraining interrupted by user")
        finally:
            # 关闭所有进度条
            epoch_pbar.close()
            print("\nTraining completed")
            print(f"Total training time: {time.time() - total_start_time:.1f}s")

def main():
    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.deterministic = True

    # Create trainer
    trainer = AlphaGoZeroTrainer(
        board_size=9,
        num_channels=32,
        num_res_blocks=6,
        num_simulations=200,
        replay_buffer_size=10000,
        batch_size=64,
        num_epochs=100,
        games_per_epoch=20,
        eval_games=10,
        checkpoint_dir='checkpoints',
        learning_rate=0.001,
        weight_decay=1e-4
    )

    # Start training
    trainer.train()


if __name__ == '__main__':
    main()