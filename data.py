"""
data.py - Dataset et Feature Engineering pour Pacman

============================================================================
ORIGINAL (GitHub template):
----------------------------------------------------------------------------
def state_to_tensor(state):
    # Feature engineering here
    return # tensor

class PacmanDataset(Dataset):
    def __init__(self, path):
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.inputs = []
        self.actions = []
        # Your code here
============================================================================
"""

import pickle
import torch
from torch.utils.data import Dataset

try:
    from pacman_module.game import Directions
except ImportError:
    from game import Directions  # type: ignore


# =============================================================================
# MAPPINGS ACTIONS <-> INDICES
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
# FEATURE EXTRACTION - NOTRE IMPLEMENTATION
# =============================================================================

def state_to_tensor(state: object) -> torch.Tensor:
    """
    Extrait 24 features normalisées d'un GameState.

    ---------- NOTRE IMPLEMENTATION ----------

    Features extraites (toutes normalisées ~[0,1]):
    - Position Pacman (2): px, py
    - Info Ghost (4): dx, dy, distance, adjacent
    - Info Food (4): n_food, dx, dy, distance
    - Géométrie maze (5): dist_north/south/east/west, is_corner
    - Score (1)
    - Danger (3): danger_level, ghost_blocks_food, escape_options
    - Actions légales (5): one-hot encoding

    Arguments:
        state: Objet GameState du moteur Pacman

    Returns:
        Tensor 1D de 24 float32 normalisés
    """

    # ---------- NOTRE CODE COMMENCE ICI ----------

    # 1. POSITION PACMAN
    px, py = state.getPacmanPosition()

    # 2. INFO GHOST (on se concentre sur le plus proche)
    ghost_positions = state.getGhostPositions()
    if ghost_positions:
        distances = [abs(px - gx) + abs(py - gy) for gx, gy in ghost_positions]
        min_idx = int(torch.argmin(torch.tensor(distances)))
        gx, gy = ghost_positions[min_idx]
        ghost_dist = float(distances[min_idx])
    else:
        gx, gy = px, py
        ghost_dist = 0.0

    dx_ghost = float(gx - px)
    dy_ghost = float(gy - py)
    ghost_adjacent = 1.0 if ghost_dist == 1.0 else 0.0

    # 3. INFO FOOD (on se concentre sur la plus proche)
    food = state.getFood()
    food_positions = food.asList()
    n_food = float(len(food_positions))

    if food_positions:
        food_dists = [abs(px - fx) + abs(py - fy) for fx, fy in food_positions]
        min_f_idx = int(torch.argmin(torch.tensor(food_dists)))
        fx, fy = food_positions[min_f_idx]
        closest_food_dist = float(food_dists[min_f_idx])
        dx_food = float(fx - px)
        dy_food = float(fy - py)
    else:
        closest_food_dist = 0.0
        dx_food = 0.0
        dy_food = 0.0

    # 4. GEOMETRIE DU LABYRINTHE
    walls = state.getWalls()
    W, H = walls.width, walls.height

    def dist_until_wall(start_x, start_y, dx, dy):
        """Distance jusqu'au mur dans une direction."""
        d = 0
        x, y = start_x, start_y
        while True:
            x += dx
            y += dy
            if not (0 <= x < W and 0 <= y < H):
                break
            if walls[x][y]:
                break
            d += 1
        return float(d)

    dist_north = dist_until_wall(px, py, 0, 1)
    dist_south = dist_until_wall(px, py, 0, -1)
    dist_east = dist_until_wall(px, py, 1, 0)
    dist_west = dist_until_wall(px, py, -1, 0)

    # Détection de coin (peu d'options de fuite = dangereux)
    free_dirs = 0
    for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
        nx, ny = px + dx, py + dy
        if 0 <= nx < W and 0 <= ny < H and not walls[nx][ny]:
            free_dirs += 1
    is_corner = 1.0 if free_dirs <= 2 else 0.0

    # 5. SCORE
    score = float(state.getScore())

    # 6. ACTIONS LEGALES (one-hot)
    legal_actions = state.getLegalPacmanActions()
    legal_flags = [1.0 if a in legal_actions else 0.0 for a in ACTIONS]

    # 7. FEATURES DE DANGER (innovation clé!)
    # - danger_level: inverse de la distance, haut quand ghost proche
    danger_level = 1.0 / (ghost_dist + 1.0)

    # - ghost_blocks_food: 1 si ghost entre pacman et food
    ghost_blocks_food = 0.0
    if ghost_positions and food_positions:
        same_direction = (dx_ghost * dx_food > 0 or dy_ghost * dy_food > 0)
        if same_direction and ghost_dist < closest_food_dist:
            ghost_blocks_food = 1.0

    # - escape_options: nombre de directions de fuite
    escape_options = sum(1.0 for a in legal_actions if a != Directions.STOP) / 4.0

    # 8. NORMALISATION ET CONSTRUCTION DU VECTEUR
    MAX_DIST = 20.0
    MAX_COORD = 20.0

    features = [
        # Position (2)
        float(px) / MAX_COORD,
        float(py) / MAX_COORD,
        # Ghost (4)
        dx_ghost / MAX_DIST,
        dy_ghost / MAX_DIST,
        ghost_dist / MAX_DIST,
        ghost_adjacent,
        # Food (4)
        n_food / 50.0,
        dx_food / MAX_DIST,
        dy_food / MAX_DIST,
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

    # ---------- FIN DE NOTRE CODE ----------


# =============================================================================
# DATASET CLASS - NOTRE IMPLEMENTATION
# =============================================================================

class PacmanDataset(Dataset):
    """
    Dataset PyTorch pour charger les données d'expert Pacman.

    Arguments:
        path: Chemin vers pacman_dataset.pkl

    Attributes:
        inputs: Liste de tensors de features
        labels: Liste d'indices d'actions
    """

    def __init__(self, path: str):
        """
        Charge le dataset et convertit tous les états en tensors.

        Arguments:
            path: Chemin vers le fichier pickle
        """
        with open(path, "rb") as f:
            data = pickle.load(f)

        self.inputs = []
        self.labels = []

        # ---------- NOTRE CODE COMMENCE ICI ----------

        for state, action in data:
            # Convertir GameState en tensor de features
            x = state_to_tensor(state)

            # Convertir action en index (skip si action inconnue)
            if action not in ACTION_TO_INDEX:
                continue
            y = ACTION_TO_INDEX[action]

            self.inputs.append(x)
            self.labels.append(y)

        # ---------- FIN DE NOTRE CODE ----------

    def __len__(self) -> int:
        """
        Retourne le nombre d'échantillons.

        Returns:
            Nombre de paires (state, action)
        """
        return len(self.inputs)

    def __getitem__(self, idx: int):
        """
        Récupère un échantillon.

        Arguments:
            idx: Index de l'échantillon

        Returns:
            Tuple (tensor_features, index_action)
        """
        return self.inputs[idx], self.labels[idx]
