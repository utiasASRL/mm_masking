# Set ROOTDIR to the root directory of the project
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
export ROOTDIR=$(dirname "$SCRIPT_DIR")

# Check whether container is running
container_state=$(docker inspect -f '{{.State.Running}}' mm_masking_$(id -u) 2>/dev/null)

if [ "$container_state" = "true" ]
then
	echo 'Container already running, joining it now.'
	docker exec -it mm_masking_$(id -u) /entrypoint.sh
else
	echo 'New container run initialized.'
	docker run -it --rm --name mm_masking_$(id -u) \
	--privileged \
	--network=host \
	--ipc=host \
	--gpus=all \
	-e DISPLAY=$DISPLAY \
	-e ROOTDIR=$ROOTDIR \
	-v /tmp/.X11-unix:/tmp/.X11-unix \
	-v ${HOME}/.Xauthority:${HOME}/.Xauthority:rw \
	-v $ROOTDIR:$ROOTDIR:rw \
	-w $ROOTDIR mm_masking_$(id -u)
fi
cd $ROOTDIR
