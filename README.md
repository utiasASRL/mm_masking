# Multi-modal Masking Project Repository
Paper: [Pointing the Way: Refining Radar-Lidar Localization Using Learned ICP Weights](https://arxiv.org/abs/2309.08731)

Dataset: [Boreas](https://www.boreas.utias.utoronto.ca/#/)

# Citation

```bibtex
@article{lisus_icra24,
  title={{Pointing the Way: Refining Radar-Lidar Localization Using Learned ICP Weights}},
  author={Lisus, Daniil and Laconte, Johann and Burnett, Keenan and Barfoot, Timothy D.},
  journal={arXiv preprint {\tt arXiv:2309.08731}},
  year={2023}
}
```

# Project Installation
## Initialize Repository
Clone this repository to a desired directory.

Enter the repository, initialize submodules, and set up `ROOTDIR` variable:

```
cd mm_masking
git submodule update --init --recursive
export ROOTDIR=$(pwd)
```

## Build Docker Container
Build the Docker container, this will take a long time. Note, `ROOTDIR` must be correctly set.
```
bash scripts/build_docker.sh
```
Enter the Docker container. This script is used to enter the container whenever any code is desired to be run. If the container is already running, the script will join the existing container. Note, running this script before everything is set up may throw some warnings (`No such file or directory`). These can be ignored until the project installation is complete.
```
bash scripts/run_docker.sh
```
## Build Packages & Set Up Virtual Environment
When inside the container, build the packages. This may take a while. The build procedure may throw warnings, which can be ignored so long as all packages are marked as finished at the end.
```
bash scripts/build_packages.sh
```
Finally, still inside the container, set up the virtual environment.
```
bash scripts/create_venv.sh
```
To source everything that has been built, run
```
bash scripts/setup_container.sh
```
This gets run every time the `run_docker.sh` script gets run and so only needs to be manually called whenever packages or the virtual environment is changed.

# Data Generation
## Downloading Boreas Data
