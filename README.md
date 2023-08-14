# mm_masking
Multi-modal Masking Project Repo


export ROOTDIR=${pwd}

git submodule update --init --recursive


bash scripts/build_docker.sh
bash scripts/run_docker.sh


bash scripts/build_packages.sh