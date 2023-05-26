import numpy as np
import gwsurrogate

sur = gwsurrogate.LoadSurrogate('NRHybSur3dq8')



chiA = [0, 0, 0]
chiB = [0, 0, 0]
dt = 0.1        # step size, Units of M
times = np.linspace(-2750, 100, 28501)
f_low = 5e-3    # initial frequency, Units of cycles/M

qs = np.linspace(1, 8,1000)


one_dim_set = []

for q in qs:
    t, h, dyn = sur(q, chiA, chiB, times=times, mode_list=[(2,2)], f_low=f_low) 
    one_dim_set.append(h[(2,2)].real)


one_dim_set = np.asarray(one_dim_set)
np.save(f"1d_q_1_to_8_{qs.size}.npy", one_dim_set)