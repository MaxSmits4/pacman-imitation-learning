"""
data.py - Dataset and Feature Engineering for Pacman
"""

import pickle
import torch
from torch.utils.data import Dataset

try:
    from pacman_module.game import Directions
except ImportError:
    from game import Directions  # type: ignore


# =============================================================================
# ACTION <-> INDEX MAPPINGS
# =============================================================================

ACTIONS = [
    Directions.NORTH,
    Directions.SOUTH,
    Directions.EAST,
    Directions.WEST,
    Directions.STOP,
]

ACTION_TO_INDEX = {action: idx for idx, action in enumerate(ACTIONS)}
INDEX_TO_ACTION = {idx: action for action, idx in ACTION_TO_INDEX.items()}


# =============================================================================
# FEATURE EXTRACTION
# =============================================================================

def state_to_tensor(state: object) -> torch.Tensor:
    """
    Extract 24 normalized features from a GameState.

    Features (all normalized ~[0,1]):
    - Pacman position (2): px, py
    - Ghost info (4): dx, dy, distance, adjacent
    - Food info (4): n_food, dx, dy, distance
    - Maze geometry (5): dist_north/south/east/west, is_corner
    - Score (1)
    - Danger (3): danger_level, ghost_blocks_food, escape_options
    - Legal actions (5): one-hot encoding

    Args:
        state: GameState object from Pacman engine

    Returns:
        1D Tensor of 24 normalized float32 values
    """

    # 1. PACMAN POSITION
    pac_pos_x, pac_pos_y = state.getPacmanPosition()

    # 2. GHOST INFO (focus on closest ghost)
    ghost_positions = state.getGhostPositions()
    if ghost_positions:
        distances = [abs(pac_pos_x - gx) + abs(pac_pos_y - gy) for gx, gy in ghost_positions]
        min_idx = int(torch.argmin(torch.tensor(distances)))
        ghost_x, ghost_y = ghost_positions[min_idx]
        ghost_dist = float(distances[min_idx])
    else:
        ghost_x, ghost_y = pac_pos_x, pac_pos_y
        ghost_dist = 0.0

    dx_ghost = float(ghost_x - pac_pos_x)
    dy_ghost = float(ghost_y - pac_pos_y)
    ghost_adjacent = 1.0 if ghost_dist == 1.0 else 0.0

    # 3. FOOD INFO (focus on closest food)
    food = state.getFood()
    food_positions = food.asList()
    n_food = float(len(food_positions))

    if food_positions:
        # Calculate Manhattan distance from Pacman to each food
        distances_to_all_foods = [abs(pac_pos_x - food_x) + abs(pac_pos_y - food_y)
                                   for food_x, food_y in food_positions]

        # Find which food is closest
        closest_food_index = int(torch.argmin(torch.tensor(distances_to_all_foods)))
        closest_food_x, closest_food_y = food_positions[closest_food_index]

        # Distance to that closest food
        closest_food_dist = float(distances_to_all_foods[closest_food_index])

        # Direction from Pacman to closest food
        direction_to_food_x = float(closest_food_x - pac_pos_x)
        direction_to_food_y = float(closest_food_y - pac_pos_y)
    else:
        # No food left
        closest_food_dist = 0.0
        direction_to_food_x = 0.0
        direction_to_food_y = 0.0

    # 4. MAZE GEOMETRY
    # Get wall grid and maze dimensions
    walls = state.getWalls()
    W, H = walls.width, walls.height

    def dist_until_wall(start_x, start_y, direction_x, direction_y):
        """
        Count how many steps Pacman can take in a direction before hitting a wall.

        Args:
            start_x, start_y: Pacman's current position
            direction_x, direction_y: Direction to check (e.g., (0, 1) = North)

        Returns:
            Number of free cells before hitting a wall or boundary
        """
        distance = 0
        current_x, current_y = start_x, start_y

        while True:
            # Move one step in the direction
            current_x += direction_x
            current_y += direction_y

            # Check if we're out of bounds
            if not (0 <= current_x < W and 0 <= current_y < H):
                break

            # Check if we hit a wall
            if walls[current_x][current_y]:
                break

            distance += 1

        return float(distance)

    # Calculate distance to walls in each cardinal direction
    dist_north = dist_until_wall(pac_pos_x, pac_pos_y, 0, 1)   # Up
    dist_south = dist_until_wall(pac_pos_x, pac_pos_y, 0, -1)  # Down
    dist_east = dist_until_wall(pac_pos_x, pac_pos_y, 1, 0)    # Right
    dist_west = dist_until_wall(pac_pos_x, pac_pos_y, -1, 0)   # Left

    # Corner detection: count how many directions are free
    # If <= 2 directions free, Pacman is in a corner (dangerous!)
    free_directions = 0
    for direction_x, direction_y in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
        neighbor_x = pac_pos_x + direction_x
        neighbor_y = pac_pos_y + direction_y

        # Check if neighbor is valid and not a wall
        if 0 <= neighbor_x < W and 0 <= neighbor_y < H and not walls[neighbor_x][neighbor_y]:
            free_directions += 1

    # 1 if in corner (<=2 exits), 0 otherwise
    is_corner = 1.0 if free_directions <= 2 else 0.0

    # 5. SCORE
    score = float(state.getScore())

    # 6. LEGAL ACTIONS (one-hot)
    legal_actions = state.getLegalPacmanActions()
    legal_flags = [1.0 if a in legal_actions else 0.0 for a in ACTIONS]

    # 7. DANGER FEATURES
    # - danger_level: inverse of distance, high when ghost is close
    danger_level = 1.0 / (ghost_dist + 1.0)

    # - ghost_blocks_food: 1 if ghost is between pacman and food
    ghost_blocks_food = 0.0
    if ghost_positions and food_positions:
        same_direction = (dx_ghost * direction_to_food_x > 0 or dy_ghost * direction_to_food_y > 0)
        if same_direction and ghost_dist < closest_food_dist:
            ghost_blocks_food = 1.0

    # - escape_options: number of escape directions
    escape_options = sum(1.0 for a in legal_actions if a != Directions.STOP) / 4.0

    # 8. NORMALIZATION AND VECTOR CONSTRUCTION
    MAX_DIST = 20.0
    MAX_COORD = 20.0

    features = [
        # Position (2)
        float(pac_pos_x) / MAX_COORD,
        float(pac_pos_y) / MAX_COORD,
        # Ghost (4)
        dx_ghost / MAX_DIST,
        dy_ghost / MAX_DIST,
        ghost_dist / MAX_DIST,
        ghost_adjacent,
        # Food (4)
        n_food / 50.0,
        direction_to_food_x / MAX_DIST,
        direction_to_food_y / MAX_DIST,
        closest_food_dist / MAX_DIST,
        # Geometry (5)
        dist_north / 10.0,
        dist_south / 10.0,
        dist_east / 10.0,
        dist_west / 10.0,
        is_corner,
        # Score (1)
        score / 500.0,
        # Danger (3)
        danger_level,
        ghost_blocks_food,
        escape_options,
        # Legal actions (5)
        *legal_flags,
    ]

    return torch.tensor(features, dtype=torch.float32)


# =============================================================================
# DATASET CLASS
# =============================================================================

class PacmanDataset(Dataset):
    """
    PyTorch Dataset for loading Pacman expert data.

    Args:
        path: Path to pacman_dataset.pkl

    Attributes:
        inputs: List of feature tensors
        labels: List of action indices
    """

    def __init__(self, path: str):
        """
        Load dataset and convert all states to tensors.

        Args:
            path: Path to pickle file
        """
        with open(path, "rb") as f:
            data = pickle.load(f)

        self.inputs = []
        self.labels = []

        for state, action in data:
            # Convert GameState to feature tensor
            x = state_to_tensor(state)

            # Convert action to index (skip if unknown action)
            if action not in ACTION_TO_INDEX:
                continue
            y = ACTION_TO_INDEX[action]

            self.inputs.append(x)
            self.labels.append(y)

    def __len__(self) -> int:
        """Returns number of samples."""
        return len(self.inputs)

    def __getitem__(self, idx: int):
        """
        Get a sample.

        Args:
            idx: Sample index

        Returns:
            Tuple (feature_tensor, action_index)
        """
        return self.inputs[idx], self.labels[idx]
