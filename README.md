# mm_masking
Multi-modal Masking Project Repo

export ROOTDIR=${pwd}
bash scripts/build_docker.sh
bash scripts/run_docker.sh


git submodule update --init --recursive

bash scripts/build_packages.sh