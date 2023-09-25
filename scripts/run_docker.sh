# Assumes that ROOTDIR is set and pointing to radar_topometric_localization root directory
container_state=$(docker inspect -f '{{.State.Running}}' mm_masking_temp 2>/dev/null)

if [ "$container_state" = "true" ]
then
	echo 'Container already running, joining it now.'
	docker exec -it mm_masking_temp /entrypoint.sh
else
	echo 'New container run initialized.'
	docker run -it --rm --name mm_masking_temp \
	--privileged \
	--network=host \
	--ipc=host \
	--gpus all \
	-e DISPLAY=$DISPLAY \
	-e ROOTDIR=$ROOTDIR \
	-v /tmp/.X11-unix:/tmp/.X11-unix \
	-v ${HOME}/.Xauthority:${HOME}/.Xauthority:rw \
	-v /raid/dli:/raid/dli:rw \
    -v /raid/krb:/raid/krb:rw \
	-v /nas/ASRL/2021-Boreas:/nas/ASRL/2021-Boreas \
	-v $ROOTDIR:$ROOTDIR:rw \
	-w $ROOTDIR mm_masking_temp
fi
cd $ROOTDIR