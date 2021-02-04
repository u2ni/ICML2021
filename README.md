# ICML2021 - On the Effectiveness of Weight-Encoded Neural Implicit 3D Shapes
A neural implicit outputs a number indicating whether the given query point in space is outside, inside, or on a surface. Many prior works have focused on _latent-encoded_ neural implicits, where latent vector encoding a specific shape is also fed as input. While affording latent-space interpolation, this comes at the cost of reconstruction accuracy for any _single_ shape.  Training a specific network for each 3D shape, a _weight-encoded_ neural implicits may forgo the latent vector and focus reconstruction accuracy on the details of a single shape. While previously considered as intermediary representation 3D scanning tasks or as a toy-problem leading up to latent-encoding tasks, weight-encoded neural implicits have not yet been taken seriously as a 3D shape representation. In this paper, we establish that weight-encoded neural implicits meet the criteria of a first-class 3D shape representation. We introduce a suite of technical contributions to improve reconstruction accuracy, convergence, and robustness when learning the signed distance field induced by a polygonal mesh --- the _de facto_ standard representation. Viewed as a lossy compression, our conversion outperforms standard techniques from geometry processing. Compared to previous latent- and weight-encoded neural implicits we demonstrate superior robustness, accuracy, scalability, and performance.

In this repo we present a comprehensive package for generating weight-encoded neural implicits from existing mesh formats and visualizing the results through a cuda or eigen based sphere marchers.

We additionally share the weight-encoded neural implicits for the Entirety of the Thingk10k dataset (https://ten-thousand-models.appspot.com/). 

## Lay of the land
  **Supplementary.mp4** --> supplementary video visualizing through animations various concepts within the paper.  
  **cudaSDFRenderer/** --> our weight-encoded neural implicit GPU accelerated renderer. Use this for realtime rendering of neural implicits.
  **neuralImplicitTools** --> tools for sampling meshes, optimizing weights, marching cubes, and visualizing results. This is a good starting point.
  **thingi10k-weight-encoded** --> The entirety of Thingi10k represented in our weight-encoded neural implicit format. (see releases for mesh version for convenience)


