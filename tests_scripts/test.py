

import numpy as np


a = np.array([-1,2,3,-1,5])
xyxy_a = np.array([
    [0,0,0,0],
    [1,1,1,1],
    [2,2,2,2],
    [3,3,3,3],
    [4,4,4,4],
])

b = np.array([5,4,3,2,-1])
xyxy_b = np.array([
    [5,5,5,5],
    [6,6,6,6],
    [7,7,7,7],
    [8,8,8,8],
    [9,9,9,9]
])

filter = a != -1
a = a[filter]
xyxy_a = xyxy_a[filter]

filter = b != -1
b = b[filter]
xyxy_b = xyxy_b[filter]

print(a, xyxy_a)
print()
print(b, xyxy_b)
print()

tracker_ids, idx_as, idx_bs = np.intersect1d(a, b, assume_unique=True, return_indices=True)
print(tracker_ids, idx_as, idx_bs)
print()
xyxy_a = xyxy_a[idx_as]
xyxy_b = xyxy_b[idx_bs]

print(tracker_ids)
print(xyxy_a)
print(xyxy_b)