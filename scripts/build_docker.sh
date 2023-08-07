# Assumes that ROOTDIR is set and pointing to radar_topometric_localization root directory
cd $ROOTDIR
docker build -t mm_masking -f Dockerfile .
cd $ROOTDIR