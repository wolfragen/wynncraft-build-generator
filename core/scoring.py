def compute_score(state, query):

    score = 0.0

    # ---- Stat contribution ----
    for i in query.relevant_stat_indices:
        if query.weights[i] != 0:
            val = state.compute_effective_stat(i)
            score += val * query.weights[i]

    # ---- Durability contribution ----
    if query.durability_weight != 0:
        score += state.durability * query.durability_weight

    return score