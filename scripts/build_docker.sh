# Assumes that ROOTDIR is set and pointing to radar_topometric_localization root directory
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
export ROOTDIR=$(dirname "$SCRIPT_DIR")

docker build -t mm_masking_$(id -u) -f Dockerfile \
    --build-arg USERID=$(id -u) \
    --build-arg GROUPID=$(id -g) \
    --build-arg USERNAME=$(whoami) \
    --build-arg HOMEDIR=${HOME} \
    --build-arg ROOTDIR=${ROOTDIR} .
cd $ROOTDIR