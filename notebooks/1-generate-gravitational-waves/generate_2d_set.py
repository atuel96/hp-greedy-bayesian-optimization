import numpy as np
import gwsurrogate

sur = gwsurrogate.LoadSurrogate('NRHybSur3dq8')




dt = 0.1        # step size, Units of M
times = np.linspace(-2750, 100, 28501)
f_low = 5e-3    # initial frequency, Units of cycles/M

qs = np.linspace(1, 8, 50)
chis_z = np.linspace(-.8, .8, 10)



one_dim_set = []
for q in qs:
    for chi_z in chis_z:
        chi = [0, 0, chi_z]
        t, h, dyn = sur(q, chi, chi, times=times, mode_list=[(2,2)], f_low=f_low) 
        one_dim_set.append(h[(2,2)].real)

one_dim_set = np.asarray(one_dim_set)
np.save("2d_q{}xchi{}_{}.npy".format(qs.size, chis_z.size, qs.size*chis_z.size),one_dim_set)
