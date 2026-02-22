import json
import os
import sys

REQ_STATS = {"strReq", "dexReq", "intReq", "defReq", "agiReq"}

def extract_metrics(recipe):
    """Normalizes recipe data: Higher value always = Better."""
    # Efficiencies for -1 slots, sorted descending
    effs = sorted([e for i, e in zip(recipe["ings"], recipe["eff"]) if i == -1], reverse=True)
    
    stats = {}
    if "stats" in recipe:
        for name, d in recipe["stats"].items():
            # Flip Reqs so that -50 is 'worse' than 0
            stats[name] = -d["min"] if name in REQ_STATS else d["max"]
                
    return {"effs": effs, "stats": stats}

def compare_recipes(m_a, m_b):
    """Returns 1 if A > B, -1 if B > A, 0 if A == B, None if incomparable."""
    a_better_or_eq = True
    b_better_or_eq = True
    a_strictly = False
    b_strictly = False
    
    # Compare Effs
    for ea, eb in zip(m_a["effs"], m_b["effs"]):
        if ea > eb: b_better_or_eq, a_strictly = False, True
        elif eb > ea: a_better_or_eq, b_strictly = False, True

    # Compare Stats
    for key in set(m_a["stats"].keys()) | set(m_b["stats"].keys()):
        va, vb = m_a["stats"].get(key, 0), m_b["stats"].get(key, 0)
        if va > vb: b_better_or_eq, a_strictly = False, True
        elif vb > va: a_better_or_eq, b_strictly = False, True
            
    if a_better_or_eq and b_better_or_eq: return 0
    if a_better_or_eq and a_strictly: return 1
    if b_better_or_eq and b_strictly: return -1
    return None

def cull_recipes(filename):
    in_path = os.path.join("data", "precalc", "full", filename)
    out_path = os.path.join("data", "precalc", "generic_cull", filename)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    total_bytes = os.path.getsize(in_path)
    bytes_processed, last_pct, total_in = 0, -1, 0
    kept_recipes, kept_metrics = [], []

    with open(in_path, "r", encoding="utf-8") as f:
        for line in f:
            bytes_processed += len(line.encode('utf-8'))
            clean = line.strip().rstrip(',')
            if clean in ('[', ']', ''): continue
            
            try: cand = json.loads(clean)
            except: continue
                
            total_in += 1
            m_cand = extract_metrics(cand)
            dominated, to_rem = False, []
            
            for i, m_kept in enumerate(kept_metrics):
                res = compare_recipes(m_cand, m_kept)
                if res == 1: to_rem.append(i)
                elif res in (-1, 0): 
                    dominated = True
                    break
                    
            if not dominated:
                for i in reversed(to_rem):
                    kept_recipes.pop(i)
                    kept_metrics.pop(i)
                kept_recipes.append(cand)
                kept_metrics.append(m_cand)

            pct = int((bytes_processed / total_bytes) * 100)
            if pct > last_pct:
                last_pct = pct
                sys.stdout.write(f"\r[{pct:>3}%] Total: {total_in} | Kept: {len(kept_recipes)}")
                sys.stdout.flush()

    # Output in flat format: [ \n {recipe}, \n {recipe} \n ]
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("[\n")
        for i, r in enumerate(kept_recipes):
            line = json.dumps(r)
            f.write(f"{line}{',' if i < len(kept_recipes)-1 else ''}\n")
        f.write("]")
    print(f"\nSaved to {out_path}")

# cull_recipes("JEWELING_META_1.json")




# Unit testing - test_all returns True currently 
class UnitTest():
    """compare_recipes(m_a, m_b): Returns 1 if A > B, -1 if B > A, 0 if A == B, None if incomparable."""
    def test_all(self):
        success = self.test_requirements() and self.test_inverse() and self.test_negative_bonuses()and self.test_incomparable_positives() and self.test_eff_tradeoff()
        return success

    def test_requirements(self):
        # A has strReq 50, B has none (0). B should dominate A.
        a = {"ings": [-1], "eff": [100], "stats": {"strReq": {"min": 50, "max": 50}}}
        b = {"ings": [-1], "eff": [100], "stats": {}}
        res = compare_recipes(extract_metrics(a), extract_metrics(b))
        return res == -1 # B is better
    
    def test_inverse(self): # same as test_requirements, but inverse A and B
        # B has strReq 50, A has none (0). A should dominate B.
        a = {"ings": [-1], "eff": [100], "stats": {}}
        b = {"ings": [-1], "eff": [100], "stats": {"strReq": {"min": 50, "max": 50}}}
        res = compare_recipes(extract_metrics(a), extract_metrics(b))
        return res == 1 # A is better

    def test_negative_bonuses(self):
        # A has -10 fire, B has none (0). B should dominate A.
        a = {"ings": [-1], "eff": [100], "stats": {"fire": {"min": -10, "max": -10}}}
        b = {"ings": [-1], "eff": [100], "stats": {}}
        res = compare_recipes(extract_metrics(a), extract_metrics(b))
        return res == -1

    def test_incomparable_positives(self):
        # One has fire, one has water. Neither should dominate.
        a = {"ings": [-1], "eff": [100], "stats": {"fire": {"max": 10}}}
        b = {"ings": [-1], "eff": [100], "stats": {"water": {"max": 10}}}
        res = compare_recipes(extract_metrics(a), extract_metrics(b))
        return res == None

    def test_eff_tradeoff(self):
        # A has better first slot, B has better second. Neither dominates.
        a = {"ings": [-1, -1], "eff": [200, 50], "stats": {}}
        b = {"ings": [-1, -1], "eff": [150, 150], "stats": {}}
        res = compare_recipes(extract_metrics(a), extract_metrics(b))
        return res == None

