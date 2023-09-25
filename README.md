Paper: [Pointing the Way: Refining Radar-Lidar Localization Using Learned ICP Weights](https://arxiv.org/abs/2309.08731)

Dataset: [Boreas](https://www.boreas.utias.utoronto.ca/#/)

# Citation

```bibtex
@article{lisus_icra24,
  title={{Pointing the Way: Refining Radar-Lidar Localization Using Learned ICP Weights}},
  author={Lisus, Daniil and Laconte, Johann and Burnett, Keenan and Barfoot, Timothy D.},
  journal={arXiv preprint {\tt arXiv:2309.08731}},
  year={2023}
}
```

# mm_masking
Multi-modal Masking Project Repo


export ROOTDIR=${pwd}

git submodule update --init --recursive


bash scripts/build_docker.sh
bash scripts/run_docker.sh


bash scripts/build_packages.sh
