# Assumes that ROOTDIR is set and pointing to mm_masking root directory
# Need to additionally set VTRSRC variable
export VTRSRC=$ROOTDIR/external/vtr3
source /opt/ros/humble/setup.bash
source ${VTRSRC}/main/install/setup.bash # source the vtr3 environment
cd $ROOTDIR/external/boreas_vtr_wrapper # go to where boreas_vtr_wrapper is located
MAKEFLAGS="-j$(($(nproc --all) / 4 + 1))" colcon build --parallel-workers 1 --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release

cd $ROOTDIR