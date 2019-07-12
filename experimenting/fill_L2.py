import numpy as np

L1 = 32
L2 = 512

P = np.ones(shape=(1, L1), dtype=np.double)
Pt = np.zeros(shape=(1, L2), dtype=np.double)

shift_idx = 0
while shift_idx + L1 <= L2:  # fill L2 length
    Pt[0, shift_idx:shift_idx + L1] = Pt[0, shift_idx:shift_idx + L1] + P

    shift_idx = shift_idx + 1

print(Pt)
