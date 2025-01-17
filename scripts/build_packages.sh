# Assumes that ROOTDIR is set and pointing to mm_masking root directory

# Rebuild vtr3
cd $ROOTDIR
source scripts/build_vtr3.sh

# Rebuild boreas_vtr_wrapper
cd $ROOTDIR
source scripts/build_boreas_vtr_wrapper.sh