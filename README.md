This project explores the principles of Geometric Machine Learning (equivariances &amp; invariances) and Topological Analysis applied to a problem of 2D Computational Fluid Dynamics.

# Introduction

## Problem Statement
We are given a 2D rectangular CFD domain of a fixed size. The left and the right boundaries are an *Inlet* and an *Outlet*, which define the flow of the liquid through the domain. The top and the bottom boundaries are fixed solid walls. A cylindrical obstacle defined by $\left((x,y),R\right)$ obstructs the flow.

<img height="250" alt="image" src="https://github.com/user-attachments/assets/236a77fe-bb6d-479a-b3c6-32b23efd9089" />
<img height="250" alt="image" src="https://github.com/user-attachments/assets/73ed32c3-0f2f-47db-9f65-cf577730f6f5" />

Our target is to develop an ML algorithm which for a given CFD domain and a given cylinder correctly builds the mesh, calculates the relevant $(u,v,p)$ field quantities at each mesh node, and predicts the time dynamics. The desired outcome is osmething like: https://drive.google.com/file/d/1tHdQvCbvCvDpkXEGaDgWqgLknotfDNpE/view

# Resources
## Dataset
- Xu, Wenzhuo, Noelia Grande Gutierrez, and Christopher McComb. "MegaFlow2D: A parametric dataset for machine learning super-resolution in computational fluid dynamics simulations." Proceedings of Cyber-Physical Systems and Internet of Things Week 2023. 2023. 100-104.
  - Paper: [https://dl.acm.org/doi/abs/10.1145/3576914.3587552](https://dl.acm.org/doi/abs/10.1145/3576914.3587552)
  - GitHub: [https://github.com/cmudrc/MegaFlow2D](https://github.com/cmudrc/MegaFlow2D)

## Relevant papers
- Cheng, S., Bocquet, M., Ding, W., Finn, T. S., Fu, R., Fu, J., ... & Arcucci, R. (2025). Machine learning for modelling unstructured grid data in computational physics: a review. Information Fusion, 103255. [https://www.sciencedirect.com/science/article/pii/S1566253525003288](https://www.sciencedirect.com/science/article/pii/S1566253525003288)
- Chen, J., E. Hachem, and J. Viquerat. "Graph neural networks for laminar flow prediction around random two-dimensional shapes." (2021). [https://hal.science/hal-03432662/](https://hal.science/hal-03432662/) 
- Pfaff, T., Fortunato, M., Sanchez-Gonzalez, A., & Battaglia, P. (2020, October). Learning mesh-based simulation with graph networks. In International conference on learning representations.
  - GitHub repo: [sites.google.com/view/meshgraphnets](https://github.com/google-deepmind/deepmind-research/tree/master/meshgraphnets)
  - Video site: [sites.google.com/view/meshgraphnets](sites.google.com/view/meshgraphnets)
  - Paper: [arxiv.org/abs/2010.03409](arxiv.org/abs/2010.03409)
- Fink, O., Nejjar, I., Sharma, V., Niresi, K. F., Sun, H., Dong, H., ... & Zhao, M. (2025). From Physics to Machine Learning and Back: Part II-Learning and Observational Bias in PHM. arXiv preprint arXiv:2509.21207. [https://arxiv.org/abs/2509.21207](https://arxiv.org/abs/2509.21207) 
 
## Lectures & Learning Materials
- Nice youtube intro: 
[Equivariant Neural Networks | Part 1/3 - Introduction](https://www.youtube.com/watch?v=2bP_KuBrXSc) | [Part 2/3 - Generalized CNNs](https://www.youtube.com/watch?v=r0xyxe31QgU) | [Part 3/3 - Transformers and GNNs](https://www.youtube.com/watch?v=RBKERHaiEKY)
	- some extra on group theory [Group theory, abstraction, and the 196,883-dimensional monster](https://www.youtube.com/watch?v=mH0oCDa74tE)
	- [UvA - An Introduction to Group Equivariant Deep Learning](https://uvagedl.github.io/)
- Equivariant neural networks  –  what, why and how ? [https://maurice-weiler.gitlab.io/blog_post/cnn-book_1_equivariant_networks/](https://maurice-weiler.gitlab.io/blog_post/cnn-book_1_equivariant_networks/)
- [**AMMI Course "Geometric Deep Learning" - Lecture 1 (Introduction) - Michael Bronstein**](https://www.youtube.com/watch?v=PtA0lg_e5nA)
- Bronstein, M. M., Bruna, J., Cohen, T., & Veličković, P. (2021). Geometric deep learning: Grids, groups, graphs, geodesics, and gauges. arXiv preprint arXiv:2104.13478. [https://arxiv.org/abs/2104.13478](https://arxiv.org/abs/2104.13478)
