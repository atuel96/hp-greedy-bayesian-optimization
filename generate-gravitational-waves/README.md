# Generate Gravitational Waves

The wave functions are generated with the surrogate model NRHybSur3dq8 [[1]](#1) using the `gwsurrogate` python library. For simplicity, in this work we only use the $(l, m) = (2, 2)$ dominant angular mode.

## How to generate gravitational waves

Before starting, you need to use Linux and install the [Conda](https://conda.io/projects/conda/en/stable/user-guide/install/download.html) package manager. If you use Windows, you can use [WSL2](https://learn.microsoft.com/en-us/windows/wsl/install).

### Part 1. Set up the environment
1. Download the [gw-environment.yml](gw-environment.yml) [YAML](https://www.freecodecamp.org/news/what-is-yaml-the-yml-file-format/) file.
2. Create the envirnment with `conda env create -f gw-environment.yml`
3. Verify everything is ok with `conda env list`. You should see an environment called `gwenv`.
4. Activate the environment with `conda activate gwenv`.
### Part 2. Use the script
With the `gwenv` environment active, you can use the [generate-gravitational-waves.py](generate-gravitational-waves.py) python script.

The first time you run the script it may take several minutes for the surrogate model to be downloaded. 

By default the script will generate a set with 100 wavefunctions equispaced in the one-dimensional parameter $q$, the mass ratio, with $q \in[1,8]$. The output is an ndarray saved in a file with [.npy extension](https://numpy.org/devdocs/reference/generated/numpy.lib.format.html).

You can check this by running:

    python generate-gravitational-waves.py

To generate a set with more waves, for example 500, run:

    python generate-gravitational-waves.py -q 500

You can run `python generate-gravitational-waves.py -h` to see the usage. 

## References
<a id="1">[1]</a>
V. Varma, S. E. Field, M. A. Scheel, J. Blackman, L. E. Kidder, and H. P. Pfeiffer, Surrogate model of hybridized numerical relativity binary black hole waveforms, [Phys. Rev. D 99, 064045 (2019).](https://journals.aps.org/prd/abstract/10.1103/PhysRevD.99.064045)