# simple utility script for turning models into meshes!

import geometry as gm
import argparse
import tensorflow as tf
import os
import numpy as np

def loadModel(modelPath, neuralKey=''):
    # LOAD THE MODEL
    #load serialized model
    if neuralKey == '':
        jsonFile = open(modelPath+'.json', 'r')
    else:
        jsonFile = open(neuralKey, 'r')

    sdfModel = tf.keras.models.model_from_json(jsonFile.read())
    jsonFile.close()
    #load weights
    sdfModel.load_weights(modelPath + '.h5')
    #sdfModel.summary()

    return sdfModel

def inferSDF(sdfModel, res):
    # create data sequences
    cubeMarcher = gm.CubeMarcher()
    inferGrid = cubeMarcher.createGrid(res)
    S = sdfModel.predict(inferGrid)
    return -S

def marchMesh(S, res):
    cubeMarcher = gm.CubeMarcher()
    inferGrid = cubeMarcher.createGrid(res)
    cubeMarcher.march(inferGrid,S)
    marchedMesh = cubeMarcher.getMesh() 
    return marchedMesh


if __name__ == "__main__":
    # this should handle folders of meshes, parallelizing the meshing to avail cores
    parser = argparse.ArgumentParser(description='Neural Implicit mesher.')
    parser.add_argument('weightPath', type=str, help="path to neural implicit to be meshed, or folder of neural implicits")
    parser.add_argument('--outputPath', type=str,default='', help='destination path of generated meshes')
    parser.add_argument('--neuralKey', type=str, default='', help='path to neural implicit architecture json (the neural key)')
    parser.add_argument('--res', type=int,default=128, help='resolution of grid used in marching cubes')
    args = parser.parse_args()

    # support both single neural implicit, and a folder
    if os.path.isdir(args.weightPath):
        trainedModels = list([f.split('.')[0] for f in os.listdir(args.weightPath) if '.h5' in f])
        trainedModels = [os.path.join(args.weightPath, m) for m in trainedModels]
    else:
        trainedModels = [args.weightPath]

    # default to same location as weight path
    if (args.outputPath == ''):
        outputPath = args.weightPath
    else:
        outputPath = args.outputPath

    for m in trainedModels:
        try:
            print("[INFO] Loading model: ", m)
            sdfModel = loadModel(m, args.neuralKey)
            print("[INFO] Inferring sdf...")
            S = inferSDF(sdfModel,args.res)
            print("[INFO] Marching cubes...")
            mesh = marchMesh(S, args.res)
            mp = os.path.join(outputPath,os.path.basename(m) + '.obj')
            print("[INFO] Saving mesh to file: ",mp )
            mesh.save(mp)
            print("[INFO] Done.")
        except Exception as e:
            print (e)
    


