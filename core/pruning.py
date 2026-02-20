from core.scoring import compute_score


def can_still_reach_requirements(state, query, bounds, best_score=None):
    """
    Pruning with:
    - Min / Max feasibility
    - Durability feasibility
    - Optional score upper bound pruning
    """

    remaining = state.max_depth - state.depth

    amp_pos = 1.0 + bounds.max_effectiveness / 100.0
    amp_neg = 1.0 + bounds.min_effectiveness / 100.0

    # -------------------------------------------------
    # Feasibility (min/max constraints)
    # -------------------------------------------------

    for i in range(query.n_stats):

        # Current raw totals
        current_raw_max = 0
        current_raw_min = 0

        for s in range(state.depth):
            current_raw_max += state.slot_raw_max[s, i]
            current_raw_min += state.slot_raw_min[s, i]

        max_gain = bounds.max_stat_gain[i]
        min_gain = bounds.min_stat_gain[i]

        # ----- MIN constraint -----
        if query.has_min_mask[i]:

            best_per_slot = max_gain * amp_pos

            if query.search_for_inversion:
                inv_case = abs(min_gain) * abs(amp_neg)
                if inv_case > best_per_slot:
                    best_per_slot = inv_case

            best_possible = (
                current_raw_max
                + remaining * best_per_slot
            )

            if best_possible < query.min_vals[i]:
                return False

        # ----- MAX constraint -----
        if query.has_max_mask[i]:

            worst_per_slot = min_gain * amp_pos

            if query.search_for_inversion:
                inv_case = -abs(max_gain) * abs(amp_neg)
                if inv_case < worst_per_slot:
                    worst_per_slot = inv_case

            worst_possible = (
                current_raw_min
                + remaining * worst_per_slot
            )

            if worst_possible > query.max_vals[i]:
                return False

    # -------------------------------------------------
    # Durability feasibility
    # -------------------------------------------------

    if query.min_durability is not None:

        best_dura = (
            state.durability
            + remaining * bounds.max_durability
        )

        if best_dura < query.min_durability:
            return False

    # -------------------------------------------------
    # Score upper bound pruning
    # -------------------------------------------------

    if best_score is not None:

        # Current exact score
        current_score = compute_score(state, query)

        # Optimistic future gain
        optimistic_gain = 0.0

        for i in range(query.n_stats):

            if query.weights[i] == 0:
                continue

            max_gain = bounds.max_stat_gain[i]

            best_per_slot = max_gain * amp_pos

            if query.search_for_inversion:
                min_gain = bounds.min_stat_gain[i]
                inv_case = abs(min_gain) * abs(amp_neg)
                if inv_case > best_per_slot:
                    best_per_slot = inv_case

            optimistic_gain += (
                remaining * best_per_slot * query.weights[i]
            )

        upper_bound_score = current_score + optimistic_gain

        if upper_bound_score <= best_score:
            return False

    return True