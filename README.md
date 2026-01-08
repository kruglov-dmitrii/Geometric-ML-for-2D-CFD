This project explores the principles of Geometric Machine Learning applied to a problem of 2D Computational Fluid Dynamics.

# Overview
I consider a 2D rectangular CFD domain of fixed size, modeling an incompressible laminar flow governed by the Navier–Stokes equations. The left and right boundaries act as an inlet and an outlet, defining the flow through the domain, while the top and bottom boundaries are fixed no-slip walls. A cylindrical obstacle obstructs the flow inside the domain. 

The goal is to develop a mesh-based machine learning model which, for a given domain, cylinder configuration, and fixed mesh, predicts the time-dependent flow velocity fields $(u,v)$ and pressure field $p$ at each mesh node.

https://github.com/user-attachments/assets/4285d496-9347-4ca8-bae8-4700f9e65465

# Data
In this work I am using the `cylinder_flow` dataset from the [DeepMind’s MeshGraphNets framework](https://github.com/google-deepmind/deepmind-research/tree/master/meshgraphnets), 

## Data structure
The dataset consists of 1000/100/100 train/validation/test samples. Each sample represents the mesh and the (time-dependent) field solutions for a particular cylinder position and size.

The data containing in each sample:
- **`sample['mesh_pos']`** — `(num_nodes, 2)`
	- Mesh node coordinates. Each row corresponds to the $(x, y)$ position of a mesh node.

   		<img width="325" alt="image" src="https://github.com/user-attachments/assets/6641ff01-3f32-42b6-8e8f-9d812c814cc6" />

- **`sample['cells']`** — `(num_cells, 3)`
	- Mesh connectivity. Each row lists the indices of the three nodes forming a triangular cell.
	 	``` python
		>> sample['cells'][5,:]
	    tensor([10, 13, 14], dtype=torch.int32)
	    ```

		<img width="325" alt="image" src="https://github.com/user-attachments/assets/fedea746-cb38-4aca-947f-4064ef499946" />

   
- **`sample['node_type']`** — `(num_nodes,1)`
	- Integer encoding of node types: 0-internal node, 4-inlet node, 5-outlet node, 6-wall nodes.
- **`sample['pressure']`**, size: `(num_timesteps, num_nodes, 1)`
	- Time-dependent pressure field.
- **`sample['velocity']`** — `(num_timesteps, num_nodes, 2)`
	- Time-dependent velocity field. The last dimension corresponds to the $x$- and $y$- velocity $(u,v)$ at the node.

		<img width="750" alt="image" src="https://github.com/user-attachments/assets/75af2b37-6b9c-4f1b-bdf6-7133ba75531c" />


## Downloading the training data
Instructions on how to download the training dataset can be found here: https://github.com/google-deepmind/deepmind-research/tree/master/meshgraphnets. I've downloaded the data via the `download_dataset.sh` script:

```bash
sh download_dataset.sh cylinder_flow $DESTINATION_FOLDER
```

<details>
  <summary>The download_dataset.sh script</summary>
	
  ```bash
#!/bin/bash
# Copyright 2020 Deepmind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Usage:
#     sh download_dataset.sh ${DATASET_NAME} ${OUTPUT_DIR}
# Example:
#     sh download_dataset.sh flag_simple /tmp/

set -e

DATASET_NAME="${1}"
OUTPUT_DIR="${2}/${DATASET_NAME}"

BASE_URL="https://storage.googleapis.com/dm-meshgraphnets/${DATASET_NAME}/"

mkdir -p ${OUTPUT_DIR}
for file in meta.json train.tfrecord valid.tfrecord test.tfrecord
do
wget -O "${OUTPUT_DIR}/${file}" "${BASE_URL}${file}"
done	
  ```
</details>

## Converting data to PyTorch format
DeepMind team used TensorFlow in their work, and so the `test`, the `valid`, and the `train` sets are downloaded in the `.tfrecord` format. 
```bash
├── meta.json
├── test.tfrecord
├── train.tfrecord
└── valid.tfrecord
```

I have converted these into `.pt` files that are easily accessible via `torch.load()`. After conversion the data looks like:
```bash
./pt_test/
├── index.json
├── sample_000000.pt
├── sample_000001.pt
├── sample_000002.pt
├── sample_000003.pt
...
```



# Resources
## Dataset
- Pfaff, T., Fortunato, M., Sanchez-Gonzalez, A., & Battaglia, P. (2020, October). Learning mesh-based simulation with graph networks. In International conference on learning representations.
  - GitHub repo: [https://github.com/google-deepmind/deepmind-research/tree/master/meshgraphnets](https://github.com/google-deepmind/deepmind-research/tree/master/meshgraphnets)
  	- **cylinder_flow** dataset from https://github.com/google-deepmind/deepmind-research/tree/master/meshgraphnets#datasets
  - Video site: [https://sites.google.com/view/meshgraphnets](https://sites.google.com/view/meshgraphnets)
  - Paper: [arxiv.org/abs/2010.03409](arxiv.org/abs/2010.03409)


## Relevant papers
- Cheng, S., Bocquet, M., Ding, W., Finn, T. S., Fu, R., Fu, J., ... & Arcucci, R. (2025). Machine learning for modelling unstructured grid data in computational physics: a review. Information Fusion, 103255. [https://www.sciencedirect.com/science/article/pii/S1566253525003288](https://www.sciencedirect.com/science/article/pii/S1566253525003288)
- Chen, J., E. Hachem, and J. Viquerat. "Graph neural networks for laminar flow prediction around random two-dimensional shapes." (2021). [https://hal.science/hal-03432662/](https://hal.science/hal-03432662/) 
- Fink, O., Nejjar, I., Sharma, V., Niresi, K. F., Sun, H., Dong, H., ... & Zhao, M. (2025). From Physics to Machine Learning and Back: Part II-Learning and Observational Bias in PHM. arXiv preprint arXiv:2509.21207. [https://arxiv.org/abs/2509.21207](https://arxiv.org/abs/2509.21207) 
- Xu, Wenzhuo, Noelia Grande Gutierrez, and Christopher McComb. "MegaFlow2D: A parametric dataset for machine learning super-resolution in computational fluid dynamics simulations." Proceedings of Cyber-Physical Systems and Internet of Things Week 2023. 2023. 100-104.
  - Paper: [https://dl.acm.org/doi/abs/10.1145/3576914.3587552](https://dl.acm.org/doi/abs/10.1145/3576914.3587552)
  - GitHub: [https://github.com/cmudrc/MegaFlow2D](https://github.com/cmudrc/MegaFlow2D)

## Lectures & Learning Materials
- Nice youtube intro: 
[Equivariant Neural Networks | Part 1/3 - Introduction](https://www.youtube.com/watch?v=2bP_KuBrXSc) | [Part 2/3 - Generalized CNNs](https://www.youtube.com/watch?v=r0xyxe31QgU) | [Part 3/3 - Transformers and GNNs](https://www.youtube.com/watch?v=RBKERHaiEKY)
	- some extra on group theory [Group theory, abstraction, and the 196,883-dimensional monster](https://www.youtube.com/watch?v=mH0oCDa74tE)
	- [UvA - An Introduction to Group Equivariant Deep Learning](https://uvagedl.github.io/)
- Equivariant neural networks  –  what, why and how ? [https://maurice-weiler.gitlab.io/blog_post/cnn-book_1_equivariant_networks/](https://maurice-weiler.gitlab.io/blog_post/cnn-book_1_equivariant_networks/)
- [**AMMI Course "Geometric Deep Learning" - Lecture 1 (Introduction) - Michael Bronstein**](https://www.youtube.com/watch?v=PtA0lg_e5nA)
- Bronstein, M. M., Bruna, J., Cohen, T., & Veličković, P. (2021). Geometric deep learning: Grids, groups, graphs, geodesics, and gauges. arXiv preprint arXiv:2104.13478. [https://arxiv.org/abs/2104.13478](https://arxiv.org/abs/2104.13478)
- Machine Learning for Computational Fluid Dynamics [https://www.youtube.com/watch?v=IXMSOSEj14Q](https://www.youtube.com/watch?v=IXMSOSEj14Q)
