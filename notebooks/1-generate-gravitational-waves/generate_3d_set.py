import numpy as np
import gwsurrogate

sur = gwsurrogate.LoadSurrogate('NRHybSur3dq8')


dt = 0.1        # step size, Units of M
times = np.linspace(-2750, 100, 28501)
f_low = 5e-3    # initial frequency, Units of cycles/M

qs = np.linspace(1, 8, 30)
chis_z = np.linspace(-.8, .8, 15)

one_dim_set = []

for q in qs:
    for chi_z1 in chis_z:
        for chi_z2 in chis_z:
            chi_1 = [0, 0, chi_z1]
            chi_2 = [0, 0, chi_z2]
            t, h, dyn = sur(q, chi_1, chi_2, times=times, mode_list=[(2,2)], f_low=f_low) 
            one_dim_set.append(h[(2,2)].real)


one_dim_set = np.asarray(one_dim_set)
np.save("3d_q{}xchi{}_{}.npy".format(qs.size, chis_z.size, qs.size*chis_z.size**2), one_dim_set)
