Please refer to: https://github.com/u2ni/ICLR2021 

In this repo we present a comprehensive package for generating neural implicits from existing mesh formats and visualizing the results through a cuda or eigen based sphere marchers.

All code can be found in... /code/
	- cudaSDFRenderer
		- executable in /build/src/ build for ubuntu 18
		- depends on cuda 10, and requires cuda device >= 6.
		- cutlass: https://github.com/NVIDIA/cutlass
		- usage can be found by running executable with -h
		- build with
		-	mkdir build && cd build && cmake .. && make
	- shapeMemory
		- our network architecture and where you can train our model on any given geometry.
		- docker file supplied gives dependencies needed to run.
		- to train on sample mesh run
			- python3 trainer.py -i bumpy-cube.obj 
		- this will train on model suppled and visualize the results


We additionally package the entire thingi10k dataset as weight-encoded neural implicits. 

see: neuralImplicitTools/src/modelmesher.py to mesh and visualize!

Dependencies:
	- HighFive: https://github.com/BlueBrain/HighFive
	- Libigl: (packaged inside already), just needs to be built.

