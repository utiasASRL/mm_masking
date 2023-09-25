# Set up VTR directory pointers
export VTRROOT=$ROOTDIR                   # This is required for some internal scripts
export VTRSRC=$ROOTDIR/external/vtr3
export VTRRROOT=$ROOTDIR/external/vtr_testing_radar
export VTRRESULT=$ROOTDIR/data/vtr_results  # POINT THIS TO WHERE YOU WANT TO STORE RESULTS
export VTRRDATA=$ROOTDIR/data/vtr_data      # POINT THIS TO DATA DIRECTORY

# Source setups
source /opt/ros/humble/setup.bash
source $VTRSRC/main/install/setup.bash # source the vtr3 environment
source $VTRRROOT/install/setup.bash

# Create directories if they don't exist
mkdir -p $VTRRESULT
mkdir -p $VTRRDATA

# Source setups
source /opt/ros/humble/setup.bash
source $VTRSRC/main/install/setup.bash # source the vtr3 environment
source $VTRRROOT/install/setup.bash

# Activate venv
source $ROOTDIR/venv/bin/activate

export OMP_NUM_THREADS=8   # used to control the number of threads the container can use
export NEPTUNE_API_TOKEN=eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI4ODZkYzJmNS1iMWY3LTRlMWYtYWNjYy0zNTFhOWJjYjNiMTQifQ==