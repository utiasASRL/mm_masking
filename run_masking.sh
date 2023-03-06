

# Load in dataset path
DATASET=${ROOTDIR}/results/loc_rosbags/
RESULTS=${ROOTDIR}/results/radar/boreas-2020-11-26-13-58

python ${ROOTDIR}/masking/load_data.py --dataset ${DATASET} --path ${RESULTS}
