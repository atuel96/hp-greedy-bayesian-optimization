import argparse
import numpy as np
import gwsurrogate

# parse arguments
parser = argparse.ArgumentParser(description="Generate Gravitational Waves")
parser.add_argument("-d", "--dimension", type=int,
                    default=1, choices=[1, 2, 3],
                    help="Dimension of parameter")
parser.add_argument("-q", "--massratio", type=int, default=100,
                    help="Number of q (mass ratio) samples")
parser.add_argument("-x", "--spinz", type=int, default=10,
                    help="Number of spin z samples")
args = parser.parse_args()

#  generate gravitational waves
sur = gwsurrogate.LoadSurrogate('NRHybSur3dq8')

dt = 0.1        # step size, Units of M
times = np.linspace(-2750, 100, 28501)
f_low = 5e-3    # initial frequency, Units of cycles/M
qs = np.linspace(1, 8, args.massratio)
chis_z = np.linspace(-.8, .8, args.spinz)

print(f"\nGravitational Waves generation started!\n")
gws = []
filename = f"{args.dimension}d-gw"
if args.dimension == 1:  # One Dimension
    chiA = chiB = [0, 0, 0]
    for q in qs:
        t, h, dyn = sur(q, chiA, chiB, times=times,
                        mode_list=[(2, 2)], f_low=f_low)
        gws.append(h[(2, 2)])
    filename += f"-q{args.massratio}.npy"
elif args.dimension == 2:
    for q in qs:
        for chi_z in chis_z:
            chi = [0, 0, chi_z]
            t, h, dyn = sur(q, chi, chi, times=times,
                            mode_list=[(2, 2)], f_low=f_low)
            gws.append(h[(2, 2)])
    filename += f"-q{args.massratio}-chi{args.spinz}-total{args.spinz*args.massratio}.npy"
elif args.dimension == 3:
    for q in qs:
        for chi_z1 in chis_z:
            for chi_z2 in chis_z:
                chi_1 = [0, 0, chi_z1]
                chi_2 = [0, 0, chi_z2]
                t, h, dyn = sur(q, chi_1, chi_2, times=times,
                                mode_list=[(2, 2)], f_low=f_low)
                gws.append(h[(2, 2)])
    filename += f"-q{args.massratio}-chi{args.spinz}-total{args.spinz**2*args.massratio}.npy"
else:
    raise ValueError("Dimension must be 1, 2 or 3.")


gws = np.asarray(gws)
np.save(filename, gws)
print("DONE!")
