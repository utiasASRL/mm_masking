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
## Odometry (Teach) and Localization (Repeat)
This pipeline is trained, tested, and validated using the Teach & Repeat (T&R) framework on the [Boreas](https://www.boreas.utias.utoronto.ca/#/) dataset. A map is constructed during the Teach pass and subsequent Repeat passes are localized against that map. This code is contained in the `vtr3` and `vtr_testing_radar` submodules.

Make sure that `setup_container.sh` has been run, as it defines all required variables and sources all required packages!

The general form of running and evaluating a test is
```Bash
bash gen_data/run_test.sh ${MODE} ${SENSOR} ${SEQUENCES}
bash gen_data/run_eval.sh ${MODE} ${SENSOR} ${SEQUENCES}
```
where `MODE = [odometry, localization]`, `SENSOR = [radar, lidar, radar_lidar]`, and `SEQUENCES` is either one (for odometry) or two (for localization) Boreas sequence names. Consider the examples below for radar_lidar, meaning using radar to localize (Repeat) against a lidar map (Teach).

Consider, as an example, the following sequences for odometry and localization.
```Bash
# Choose a Teach (ODO_INPUT) and Repeat (LOC_INPUT) run from boreas dataset
ODO_INPUT=boreas-2020-11-26-13-58
LOC_INPUT=boreas-2021-01-26-10-59
```
Note, it is not required to define these variables, as you can input the sequence name as an argument directly. If it is desired to do a localization test, it is first required that an odometry result is generated for the sequence against which a localization attempt is desired.

Run and evaluate a single lidar odometry test using
```Bash
bash gen_data/run_test.sh odometry lidar ${ODO_INPUT}
bash gen_data/run_eval.sh odometry lidar ${ODO_INPUT}
```
Rename the resulting `data/vtr_results/lidar` directory to `data/vtr_results/radar_lidar`. 

Next, run and evaluate a single radar localization test using
```Bash
bash gen_data/run_test.sh localization radar_lidar ${ODO_INPUT} ${LOC_INPUT}
bash gen_data/run_eval.sh localization radar_lidar ${ODO_INPUT}
```

Note, that the evaluation scripts both only take in an odometry sequence. This is because the output of a localization run against a map constructed from an odometry sequence is stored under the odometry sequence result subfolder. The evaluation script evaluates all localization sequences contained within the odometry sequence subfolder at the same time.

This code has been tested with the sequences collected on and before 2021-09-14. There may be difficulties running it with newer boreas sequences, as the sensor configurations may have changed.

## Running Experiments in Parallel

Assuming you want to run odometry or localization for multiple test sequences in parallel, it is possible to do so by running
```Bash
bash gen_data/run_parallel_test.sh ${MODE} ${SENSOR}
```
where `MODE = [odometry, localization]`, `SENSOR = [radar, lidar, radar_lidar]`. This script runs all tests, either odometry or localization, and evaluates them afterwards. Note, SEQUENCES are not provided as an input for this script, as the specific list of sequences desired to be tested in parallel must be set inside of the `run_parallel_test.sh` script file. Consider the examples below for radar_lidar localization.

```Bash
bash gen_data/run_parallel_test.sh localization radar_lidar
```
Note, running this script assumes that the `REFERENCE` sequence, set inside of run_parallel_test.sh, has an already completed odometry test.

You can monitor the progress of each test by going to the log file of each test.

The log file should be located at

`${VTRRESULT}/${SENSOR}/${ODO_INPUT}/${ODO_INPUT}/<some name based on time>.log`

for odometry and at

`${VTRRESULT}/${SENSOR}/${ODO_INPUT}/${LOC_INPUT}/<some name based on time>.log`

for localization, where `${VTRRESULT}` is set in `setup_container.sh`. After the evaluation of the tests is complete, you should see the output in the terminal. Various other results can be found in the `${VTRRESULT}` directory.

## Installing and Running a Test Sequence
This repository is set up to run with Boreas sequences, which can be installed from https://www.boreas.utias.utoronto.ca/#/download. To facilitate downloading the sequences, the Docker image is set up with the necessary AWS CLI. A script to install a test sequence (approx. 5 Gb radar data) is included to facilitate verifying that all installation has completed sucessfully. This script will install a test radar sequence in the `data` folder. To download the sequence, run

```Bash
bash scripts/dl_boreas_test.sh
```

After the sequence has downloaded, this may take a while depending on your download speed, you can run a radar odometry test on the sequence through

```Bash
bash gen_data/run_test.sh odometry radar boreas-2020-11-26-13-58
```

The expected output, assuming nothing has been changed from the default configuration file, should be similar to the following

```Shell
WARNING [boreas_odometry.cpp:149] [test] Found 4142 radar data
WARNING [boreas_odometry.cpp:169] [test] Loading radar frame 0 with timestamp 1606417097528152000
WARNING [boreas_odometry.cpp:169] [test] Loading radar frame 1 with timestamp 1606417097778155000
WARNING [odometry_icp_module.cpp:553] [radar.odometry_icp] T_m_r is:     0.007637     0.015923 -1.07134e-05  2.18899e-06  8.45092e-08   0.00219319
WARNING [odometry_icp_module.cpp:554] [radar.odometry_icp] w_m_r_in_r is:   -0.0214213   -0.0636917  4.28452e-05 -7.50626e-06  1.60892e-07  -0.00437644
WARNING [boreas_odometry.cpp:169] [test] Loading radar frame 2 with timestamp 1606417098028164000
WARNING [odometry_icp_module.cpp:553] [radar.odometry_icp] T_m_r is:    0.0194236    0.0177124  0.000548001 -4.29419e-05 -1.01629e-05   0.00356502
WARNING [odometry_icp_module.cpp:554] [radar.odometry_icp] w_m_r_in_r is:  0.00215017 -0.00707829 -0.00223349 0.000172793 3.52718e-05   0.0107883
```

Consult the [Boreas download page](https://www.boreas.utias.utoronto.ca/#/download) and the example download script to download additional sequences. Remember that localization can only be run once an odometry test has been run on a different sequence.

## Using Teach and Repeat Data
The training, validation, and testing code runs extracts data directly from the `vtr_results` result directories using the `vtr3_python` submodule. Every time that ICP is run in the T&R framework (any time a sensor measurement from a repeat sequence is localized against a submap constructed during the repeat sequence during a `run_test.sh localization` script execution), the raw and processed (motion corrected) pointclouds from the repeat sequence frame and the submap against which the pointcloud is localized against are saved. The raw pointcloud is simply the original BFAR-extracted pointcloud from the raw radar scan. To ensure that all data is saved, the following configuration parameters must be set in the `gen_data/config` directory for the repeat run:

1. `save_raw_point_cloud` should be true
2. `odometry/mapping/max_translation` and `odometry/mapping/max_rotation` should both be set to 0

For the purposes of the [Pointing the Way](https://arxiv.org/abs/2309.08731) paper, this only needs to be set in the `radar_lidar_config.yaml` file, as the `lidar_config.yaml` file is used for map generation only. 

Below is an example of a directory contained within `vtr_results`. This directory contains localization results for a `radar_lidar` set up. The reference trajectory is `boreas-2020-11-26-13-58`, with the odometry (teach) results contained in the `radar_lidar/boreas-2020-11-26-13-58/boreas-2020-11-26-13-58` directory and the localization (repeat) results generated from `boreas-2020-12-04-14-00` contained in the `radar_lidar/boreas-2020-11-26-13-58/boreas-2020-12-04-14-00` directory. We can see that we succesfully collected the raw pointcloud by the presence of the `radar_raw_point_cloud` directory. A similar directory structure should be present for each localization result.

<img src="https://github.com/lisusdaniil/mm_masking/assets/26841447/4bf9ab01-1fbc-4cf7-a4be-69e26b222956" width="300">

# Training
