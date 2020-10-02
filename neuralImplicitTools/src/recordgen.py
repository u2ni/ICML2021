'''
Tool for generating tf.data records of point samples for a given set of meshes
These tf.data records can be consumed by our training pipeline :)

This tool is used for mass generation of training data for our neural implicit format. 
'''


import sys
import os
import argparse
from tqdm import tqdm

src_path = os.path.abspath(os.path.join('../'))
if src_path not in sys.path:
    sys.path.append(src_path)

igl_path = os.path.abspath(os.path.join('../../submodules/libigl/python/'))
if igl_path not in sys.path:
    sys.path.append(igl_path)

import geometry as gm
import tensorflow as tf
import numpy as np
from multiprocessing import Pool

def sampleMesh(paths):
    meshPath = paths[0]
    recordPath = paths[1]
    print("[INFO] Generating Data for mesh: {}".format(meshPath))
    # open mesh
    mesh = gm.Mesh(meshPath = meshPath)
    # generate sample queries 
    sdf = gm.SDF(mesh)
    cubeMarcher = gm.CubeMarcher()
    # query for train data
    print("[INFO] sampling {} training points".format(args.numPoints))
    sampler = gm.PointSampler(mesh, ratio = args.randomRatio, std = args.std, verticeSampling=False)
    queries = sampler.sample(args.numPoints)
    print("[INFO] inferring sdf of training points...")
    S = sdf.query(queries)
    trainSDF = np.append(queries,S, axis=1)

    valSDF = None
    if args.validationRes > 0:
        # query for validation data
        print("[INFO] sampling {} validation points".format(args.validationRes**3))
        queries = cubeMarcher.createGrid(args.validationRes)
        print("[INFO] inferring sdf of validation points...")
        S = sdf.query(queries)
        valSDF = np.append(queries,S, axis=1)

    print("[INFO] writing data to npz")

    np.savez_compressed(recordPath, train=trainSDF, validation=valSDF if not valSDF is None else [])

if __name__ == "__main__":
    # this should handle folders of meshes, parallelizing the meshing to avail cores
    parser = argparse.ArgumentParser(description='given a mesh, produce sample')
    parser.add_argument('meshPath')
    parser.add_argument('recordPath')
    parser.add_argument('--validationRes', default=128, type=int)
    parser.add_argument('--randomRatio', default=0.1,type =float,help='ratio of points on surface to random')
    parser.add_argument('--numPoints', default=10**7, type=int, help='number of points to sample')
    parser.add_argument('--std', default=0.01, type=float, help='std of normal dist we sample from surface')
    args = parser.parse_args()

    meshes = [f for f in os.listdir(args.meshPath)]

    p = Pool(6)
    
    jobs = [(os.path.join(args.meshPath, m), os.path.join(args.recordPath, os.path.splitext(m)[0] + '_sdf.npz')) for m in meshes]
    print(jobs)
    p.map(sampleMesh, jobs)



