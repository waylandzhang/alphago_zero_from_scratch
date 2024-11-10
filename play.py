import os
import sys
import logging

from scipy import ndimage

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
import torch
from src.mcts import MCTS, GoNeuralNetwork
from src import environment, govars, state_utils

# 设置日志
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

BOARD_SIZE = 9
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# 加载模型
checkpoint = torch.load('checkpoint_epoch_11_20241027_144834.pt', map_location='cpu')
model = GoNeuralNetwork(BOARD_SIZE, num_res_blocks=6)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# 初始化MCTS
mcts = MCTS(
    model=model,
    board_size=BOARD_SIZE,
    num_simulations=400,
    device=DEVICE
)


def board_to_state(board, current_player='black'):
    """将前端的棋盘格式转换为模型所需的状态格式"""
    state = np.zeros((govars.NUM_CHNLS, BOARD_SIZE, BOARD_SIZE))

    logger.debug(f"Converting board to state for player: {current_player}")
    logger.debug(f"BLACK channel index: {govars.BLACK}")
    logger.debug(f"WHITE channel index: {govars.WHITE}")

    # 设置黑白棋子位置
    for i in range(BOARD_SIZE):
        for j in range(BOARD_SIZE):
            if board[i][j] == 1:  # 黑子
                state[govars.BLACK, i, j] = 1
                logger.debug(f"Set BLACK stone at ({i}, {j})")
            elif board[i][j] == 2:  # 白子
                state[govars.WHITE, i, j] = 1
                logger.debug(f"Set WHITE stone at ({i}, {j})")

    # 设置当前玩家
    state[govars.TURN_CHNL] = 1 if current_player == 'black' else 0
    logger.debug(f"Set turn channel to: {state[govars.TURN_CHNL]} for player {current_player}")

    return state


def state_to_board(state):
    """将模型的状态格式转换为前端所需的棋盘格式"""
    board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=int)

    logger.debug(f"Converting state to board")
    logger.debug(f"Turn channel value: {state[govars.TURN_CHNL].max()}")

    # 分别处理黑子和白子，确保不会相互覆盖
    black_positions = (state[govars.BLACK] == 1)
    white_positions = (state[govars.WHITE] == 1)

    if np.any(black_positions & white_positions):
        logger.error("Found overlapping positions!")
        logger.debug(f"Overlapping positions: {np.where(black_positions & white_positions)}")

    # 设置棋子
    board[black_positions] = 1  # 黑子
    board[white_positions] = 2  # 白子

    logger.debug("Final board positions:")
    logger.debug(f"Black stones at: {np.where(black_positions)}")
    logger.debug(f"White stones at: {np.where(white_positions)}")

    return board.tolist()

# 辅助函数
def find_group(state, position, color_channel):
    """
    使用深度优先搜索找出与指定位置相连的所有同色棋子

    Args:
        state: 游戏状态数组
        position: (x, y) 起始位置
        color_channel: govars.BLACK 或 govars.WHITE

    Returns:
        set: 包含所有相连棋子坐标的集合
    """
    board_size = state.shape[1]
    visited = set()
    stack = [position]

    while stack:
        current = stack.pop()
        if current not in visited:
            visited.add(current)
            x, y = current

            # 检查四个方向
            directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
            for dx, dy in directions:
                next_x, next_y = x + dx, y + dy

                # 检查边界
                if (0 <= next_x < board_size and
                        0 <= next_y < board_size and
                        state[color_channel, next_x, next_y] == 1):
                    stack.append((next_x, next_y))

    return visited


def has_liberties(state, position):
    """
    检查指定位置的棋子所在的整个棋组是否有气

    Args:
        state: 游戏状态数组
        position: (x, y) 要检查的位置

    Returns:
        bool: 如果棋组有气则返回True，否则返回False
    """
    x, y = position
    board_size = state.shape[1]

    # 确定棋子颜色
    if state[govars.BLACK, x, y] == 1:
        color_channel = govars.BLACK
    elif state[govars.WHITE, x, y] == 1:
        color_channel = govars.WHITE
    else:
        return True  # 空位置总是有气的

    # 找出整个棋组
    group = find_group(state, position, color_channel)

    # 检查棋组的每个棋子的邻接位置
    for stone_x, stone_y in group:
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        for dx, dy in directions:
            next_x, next_y = stone_x + dx, stone_y + dy

            # 检查边界
            if (0 <= next_x < board_size and
                    0 <= next_y < board_size):
                # 如果找到一个空位，说明有气
                if (state[govars.BLACK, next_x, next_y] == 0 and
                        state[govars.WHITE, next_x, next_y] == 0):
                    return True

    return False


def check_all_groups(state):
    """
    检查棋盘上所有棋组的气

    Args:
        state: 游戏状态数组

    Returns:
        dict: 包含需要移除的黑白棋子位置的字典
    """
    board_size = state.shape[1]
    to_remove = {
        'black': set(),
        'white': set()
    }

    # 检查所有位置
    for x in range(board_size):
        for y in range(board_size):
            position = (x, y)

            # 检查黑子
            if state[govars.BLACK, x, y] == 1:
                if not has_liberties(state, position):
                    group = find_group(state, position, govars.BLACK)
                    to_remove['black'].update(group)

            # 检查白子
            elif state[govars.WHITE, x, y] == 1:
                if not has_liberties(state, position):
                    group = find_group(state, position, govars.WHITE)
                    to_remove['white'].update(group)

    return to_remove


def check_captures(state):
    """
    检查并执行提子

    Args:
        state: 游戏状态数组 (channels, height, width)

    Returns:
        tuple: (更新后的状态, 黑方提子数, 白方提子数)
    """
    new_state = state.copy()
    board_size = state.shape[1]

    # 定义2D结构元素用于连通性分析
    struct = np.array([[0, 1, 0],
                       [1, 1, 1],
                       [0, 1, 0]], dtype=bool)

    # 分别处理黑子和白子
    for color in [govars.WHITE, govars.BLACK]:
        # 获取当前颜色的棋子位置并转换为布尔数组
        pieces = new_state[color].astype(bool)

        # 标记连通区域
        labeled_groups, num_groups = ndimage.label(pieces, structure=struct)

        # 检查每个棋组
        for group_idx in range(1, num_groups + 1):
            group_mask = (labeled_groups == group_idx)

            # 获取这个棋组邻接的空位
            dilated = ndimage.binary_dilation(group_mask, structure=struct)

            # 计算所有棋子的位置（转换为布尔数组）
            all_pieces = (new_state[govars.BLACK].astype(bool) |
                          new_state[govars.WHITE].astype(bool))

            # 找出邻接的空位
            neighbors = dilated & ~all_pieces

            # 如果没有气（邻接空位），移除这个棋组
            if not np.any(neighbors):
                new_state[color][group_mask] = 0

    # 计算提子数
    black_stones = np.sum(new_state[govars.BLACK])
    white_stones = np.sum(new_state[govars.WHITE])

    return new_state, black_stones, white_stones


@app.route('/api/go/move', methods=['POST'])
def make_move():
    try:
        data = request.json
        board = data['board']
        current_player = data['currentPlayer']

        logger.debug(f"=== Starting move processing ===")
        logger.debug(f"Current player: {current_player}")

        # 将前端棋盘转换为模型状态
        state = board_to_state(board, current_player)

        # 使用MCTS获取AI的动作
        action_probs = mcts.run(state)
        action = int(np.argmax(action_probs))
        x, y = action // BOARD_SIZE, action % BOARD_SIZE

        # 执行AI的动作
        new_state = environment.next_state(state, action)

        # 检查提子
        new_state, black_stones, white_stones = check_captures(new_state)

        # 转换回前端格式
        new_board = state_to_board(new_state)

        # 确定下一个玩家
        next_player = 'white' if current_player == 'black' else 'black'

        return jsonify({
            'gameState': {
                'board': new_board,
                'currentPlayer': next_player,
                'capturedStones': {
                    'black': int(black_stones),
                    'white': int(white_stones)
                },
                'lastMove': {'x': x, 'y': y}
            }
        })

    except Exception as e:
        logger.error(f"Error in make_move: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 400

@app.route('/api/go/reset', methods=['POST'])
def reset_game():
    try:
        # 初始化新的游戏状态
        state = environment.init_state(BOARD_SIZE)
        board = state_to_board(state)

        return jsonify({
            'board': board,
            'currentPlayer': 'black'  # 确保游戏开始时是黑方
        })

    except Exception as e:
        logger.error(f"Error in reset_game: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)