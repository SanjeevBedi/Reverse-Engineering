import numpy as np

u = np.load('unified_summary.npy')
print(f'Total vertices in unified summary: {len(u)}')
print('\nLast 5 vertices:')
for i in range(110, 115):
    print(f'V{i}: ({u[i,1]:.6f}, {u[i,2]:.6f}, {u[i,3]:.6f})')

print('\nOriginal solid should have 114 vertices (indices 0-113)')
print('But unified summary has 115 vertices (indices 0-114)')
print('\nThis means one extra vertex was added during reconstruction!')
