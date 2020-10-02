We present a comprehensive package for generating neural implicits from existing mesh formats and visualizing the results through a cuda or eigen based sphere marchers.

All code can be found in... /code/
	- cudaSDFRenderer
		- executable in /build/src/ build for ubuntu 18
		- depends on cuda 10, and requires cuda device >= 6.
		- cutlass: https://github.com/NVIDIA/cutlass
		- usage can be found by running executable with -h
		- build with
		-	mkdir build && cd build && cmake .. && make
	- sdfSampler
		- a simple tool for generating training samples on mass
	- SDFViewer
		- a eigen based inference engine for generating marched meshes from neural implicits.
		- 
	- shapeMemory
		- our network architecture and where you can train our model on any given geometry.
		- docker file supplied gives dependencies needed to run.
		- to train on sample mesh run
			- python3 trainer.py -i bumpy-cube.obj 
		- this will train on model suppled and visualize the results


We addiontally package all renders of neural implicits if generating is too cumbersome.
	- videoFrames 
		- closer look at all the images that went into attached video
	- non-manifold
		- renders and neural geometries of the hotdog and swat man
	- comparisonFigure
		- all data for the comparison figure in paper comparing decimated to uniform sdf to 
	- thingi10k
		- we have converted a subset to mesh via marching cubes for easy viewing.
		- we have also included the neural geometries of the entire thingi10k dataset.

Dependencies:
	- HighFive: https://github.com/BlueBrain/HighFive
	- Libigl: (packaged inside already), just needs to be built.

