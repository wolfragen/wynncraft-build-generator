import numpy as np


class Query:
    """
    Main search class, used everywhere else.
    
    Stores all useful information to be accessed easily afterward.
    Contains : 
        - min/max values
        - weights
        - relevant stat indices
    along with other things.
    """

    __slots__ = (
        "n_stats",
        "min_vals",
        "max_vals",
        "has_min_mask",
        "has_max_mask",
        "weights",
        "search_for_inversion",
        "algorithm",
        "item_type",
        "min_durability",
        "durability_weight",
        "relevant_stat_indices"
    )

    def __init__(
        self,
        n_stats: int,
        min_vals: np.ndarray,
        max_vals: np.ndarray,
        has_min_mask: np.ndarray,
        has_max_mask: np.ndarray,
        weights: np.ndarray,
        search_for_inversion: bool,
        algorithm: str,
        item_type: str | None,
        min_durability: int | None,
        durability_weight: float,
    ):
        self.n_stats = n_stats

        self.min_vals = min_vals
        self.max_vals = max_vals

        self.has_min_mask = has_min_mask
        self.has_max_mask = has_max_mask

        self.weights = weights

        self.search_for_inversion = search_for_inversion
        self.algorithm = algorithm
        self.item_type = item_type

        self.min_durability = min_durability
        self.durability_weight = durability_weight
        
        # First check to see if stat is relevant
        # Used in loops instead of looping over all stats id, only loop over relevant stats
        self.relevant_stat_indices = np.where(self.has_min_mask | self.has_max_mask | (self.weights != 0))[0]


def build_query(
    user_query: dict,
    stat_index: dict,
    search_for_inversion: bool,
    algorithm: str,
    item_type: str | None = None,
) -> Query:
    """
    Build Query from user input.

    user_query example:
    {
        "gSpd": {"min": 1, "weight": 1.0},
        "gXp": {"min": -10, "weight": 0.35},
        "durability": {"min": 70, "weight": 0.0001},
    }
    """

    n_stats = len(stat_index)

    min_vals = np.zeros(n_stats, dtype=np.int32)
    max_vals = np.zeros(n_stats, dtype=np.int32)

    # Those two masks are here to filter ingredients. 
    # For example, we will keep positive spell dmg if has_min_mask[spell_dmg_id] == True
    has_min_mask = np.zeros(n_stats, dtype=np.bool_)
    has_max_mask = np.zeros(n_stats, dtype=np.bool_)

    weights = np.zeros(n_stats, dtype=np.float32)

    min_durability = None
    durability_weight = 0.0

    for stat_name, cfg in user_query.items(): # Checks which stats are of interest

        # ---- Durability ----
        if stat_name == "durability":

            if "min" in cfg:
                min_durability = int(cfg["min"])

            if "weight" in cfg:
                durability_weight = float(cfg["weight"])

            continue

        # ---- Normal stats ----
        if stat_name not in stat_index:
            raise ValueError(f"Unknown stat: {stat_name}")

        idx = stat_index[stat_name]

        if "min" in cfg:
            min_vals[idx] = int(cfg["min"])
            has_min_mask[idx] = True # Used to filter ingredients

        if "max" in cfg:
            max_vals[idx] = int(cfg["max"])
            has_max_mask[idx] = True # Used to filter ingredients

        if "weight" in cfg:
            weights[idx] = float(cfg["weight"]) # Score computation

    return Query(
        n_stats=n_stats,
        min_vals=min_vals,
        max_vals=max_vals,
        has_min_mask=has_min_mask,
        has_max_mask=has_max_mask,
        weights=weights,
        search_for_inversion=search_for_inversion,
        algorithm=algorithm,
        item_type=item_type,
        min_durability=min_durability,
        durability_weight=durability_weight,
    )























