## This script assumes the following environment variables are set:
##   ROOTDIR VTRRROOT VTRRDATA VTRRESULT
## These are all set automatically using the setup_container.sh script!
## example usage: bash run_parallel_test.sh localization radar

# USER INPUT: SELECT THE SEQUENCES YOU WISH TO TEST IN PARALLEL FOR EITHER MODE
if [ "$1" = "odometry" ]; then
    # Odometry sequences, SET THESE YOURSELF
    SEQUENCES=(
    'boreas-2020-11-26-13-58'
    'boreas-2020-12-04-14-00'
    'boreas-2021-01-26-10-59'
    'boreas-2021-02-09-12-55'
    'boreas-2021-03-09-14-23'
    'boreas-2021-06-29-18-53'
    'boreas-2021-09-08-21-00'
    )
else
    # Odometry reference for localization, SET THIS YOURSELF
    REFERENCE='boreas-2020-11-26-13-58'
    # Localization sequences, SET THESE YOURSELF
    SEQUENCES=(
    # Train sequences
    #'boreas-2020-12-01-13-26'
    #'boreas-2020-12-18-13-44'
    #'boreas-2021-02-02-14-07'
    #'boreas-2021-03-02-13-38'
    #'boreas-2021-03-30-14-23'
    #'boreas-2021-04-08-12-44'
    #'boreas-2021-04-20-14-11'
    #'boreas-2021-04-29-15-55'
    #'boreas-2021-05-06-13-19'
    #'boreas-2021-06-17-17-52'
    #'boreas-2021-08-05-13-34'
    #'boreas-2021-09-07-09-35'

    # Validation sequences
    #'boreas-2021-01-15-12-17'
    #'boreas-2021-03-23-12-43'
    #'boreas-2021-04-13-14-49'
    #'boreas-2021-06-03-16-00'

    
    # Test sequences
    #'boreas-2020-12-04-14-00'
    #'boreas-2021-01-26-10-59'
    #'boreas-2021-02-09-12-55'
    #'boreas-2021-03-09-14-23'
    #'boreas-2021-06-29-18-53'
    #'boreas-2021-09-08-21-00'
    )

    # Sequences that didnt work:
    # 'boreas-2021-05-13-16-11' # Doesnt converge
    # 'boreas-2021-01-19-15-08' # Malformed graph? But error is good, maybe worth retrying
fi

# Get arguments
MODE=$1         # [odometry, localization]
SENSOR=$2       # [radar, lidar, radar_lidar]

# Set results subfolder, VTRRESULT is set in setup_container.sh
export VTRRRESULT=${VTRRESULT}/${SENSOR}
mkdir -p ${VTRRRESULT}

# maximum number of jobs running in parallel
GROUPSIZE=6

SCRIPT="${ROOTDIR}/gen_data/run_test.sh"
EVAL_SCRIPT="${ROOTDIR}/gen_data/run_eval.sh"

declare -A pids

# Run tests in parallel
for seq in ${SEQUENCES[@]}; do
    if [ "$1" = "odometry" ]; then
        echo "Executing command: bash $SCRIPT $1 $2 $seq &>/dev/null &"
        ### command to execute
        bash $SCRIPT $1 $2 $seq &>/dev/null &
    else
        echo "Executing command: bash $SCRIPT $1 $2 $REFERENCE $seq &>/dev/null &"
        ### command to execute
        bash $SCRIPT $1 $2 $REFERENCE $seq &>/dev/null &
    fi

    pids[${seq}]=$!
    # wait for all pids to finish if reached group size
    if [[ ${#pids[@]} -ge ${GROUPSIZE} ]]; then
    for key in ${!pids[@]}; do
        wait ${pids[${key}]}
        echo "Process ${key} finished with return code ${?}"
        unset pids[${key}]
    done
    fi
done

for key in ${!pids[@]}; do
  wait ${pids[${key}]}
  echo "Process ${key} finished with return code ${?}"
  unset pids[${key}]
done

# Evaluate results from tests
if [ "$1" = "odometry" ]; then
    for seq in ${SEQUENCES[@]}; do
    echo "Executing command: bash $EVAL_SCRIPT $1 $2 $seq"
    bash $EVAL_SCRIPT $1 $2 $seq
    done
else
    echo "Executing command: bash $EVAL_SCRIPT $1 $2 $REFERENCE"
    bash $EVAL_SCRIPT $1 $2 $REFERENCE
fi