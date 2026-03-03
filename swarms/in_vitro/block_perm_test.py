import numpy as np
from itertools import combinations

evolved_values = [0.0003091697581112385, 0.000458647434577226, 0.0005222458262872598, 0.0004623945522890873, 0.0004129259097270477, 0.0004251285419835673, 0.00042599839381655757, 0.0006460171586614314, 0.0004678632666342659, 0.0004911205436934899, 0.0005071207132513664, 0.00041081230816837893]
random_values = [0.0003540425644657476, 0.0002921903233604821, 0.0003598021392960814, 0.00040819958492935, 0.00043901926510276013, 0.0004976298074302433, 0.0004436134775995692, 0.0003851312231347264, 0.0003836608445227878, 0.00036125667342933726, 0.0004331818510773321, 0.0003642963233336904]
# ── Enter data here ──────────────────────────────────────────────────────
# Each group has 4 blocks of 3 observations.
group_a = np.array(evolved_values)#np.array([0.000354, 0.000439,0.000385,0.000364,0.000354, 0.000439,0.000385,0.000364,0.000354, 0.000439,0.000385,0.000364
    # replace with  12 values, e.g.: 3.1, 4.7, 2.9, ...
# ])

group_b = np.array(random_values)#np.array([0.000459,0.000425,0.000468,0.000491,0.000354, 0.000439,0.000385,0.000364,0.000354, 0.000439,0.000385,0.000364
    # replace with 12 values
# ])#
# ─────────────────────────────────────────────────────────────────────────────

# assert len(group_a) == 12 and len(group_b) == 12, "Each group must have exactly 12 values."

# Reshape into blocks of 3: shape (4, 3)
blocks_a = group_a.reshape(4, 3)
blocks_b = group_b.reshape(4, 3)

# Pool all 8 blocks together
all_blocks = np.vstack([blocks_a, blocks_b])  # shape (8, 3)
n_blocks = len(all_blocks)                     # 8
n_a = len(blocks_a)                            # 4


def test_statistic(idx_a):
    """Difference in means between the two block-assigned groups."""
    idx_b = [i for i in range(n_blocks) if i not in idx_a]
    mean_a = all_blocks[list(idx_a)].mean()
    mean_b = all_blocks[idx_b].mean()
    return mean_a - mean_b


# ── Exact permutation: enumerate all C(8,4) = 70 ways to assign 4 blocks to group A
observed_stat = test_statistic(range(n_a))

null_distribution = []
for idx_a in combinations(range(n_blocks), n_a):
    null_distribution.append(test_statistic(idx_a))

null_distribution = np.array(null_distribution)

# Two-sided p-value: proportion of null statistics at least as extreme as observed
p_value = np.mean(np.abs(null_distribution) >= np.abs(observed_stat))

# ── Results ───────────────────────────────────────────────────────────────────
print("=" * 50)
print("Block Permutation Test Results")
print("=" * 50)
print(f"  Group A:  mean = {group_a.mean():.4f},  n = {len(group_a)}")
print(f"  Group B:  mean = {group_b.mean():.4f},  n = {len(group_b)}")
print(f"  Observed difference in means: {observed_stat:.4f}")
print(f"  Total permutations:           {len(null_distribution)} (exact)")
print(f"  p-value (two-sided):          {p_value:.4f}")
print("=" * 50)

if p_value < 0.05:
    print("  → p < 0.05: the difference is statistically significant.")
else:
    print("  → p ≥ 0.05: no statistically significant difference detected.")
