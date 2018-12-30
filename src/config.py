# MCTS
C_PUCT = 2**0.5
EPSILON = 0.25  # P = (1 - EPSILON) * P + EPSILON * noise
ALPHA = 0.03  # Dirichlect noise
VIRTUAL_DISCOUNT = 0.8
DEFAULT_PARALLEL_NUM = 8  # Faster at the cost of thinking depth.

# Game
BOARD_SHAPE = (9, 9)
NUM_CONNECTED_TO_WIN = 5
DEFAULT_GUIDE_ENABLED = False  # This option is slow.

# Network
L2_PENALTY = 3e-3
NUM_RESIDUAL_BLOCKS = 3
NUM_NETWORK_UNITS = 32
