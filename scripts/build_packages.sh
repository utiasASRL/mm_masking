# Assumes that ROOTDIR is set and pointing to mm_masking root directory

# Rebuild vtr3
cd $ROOTDIR
source scripts/build_vtr3.sh

# Rebuild vtr_testing_radar
cd $ROOTDIR
source scripts/build_vtr_testing_radar.sh