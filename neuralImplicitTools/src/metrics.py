# simple utility script for turning models into meshes!

import geometry as gm
import argparse
import tensorflow as tf
import os
import numpy as np
import csv

def loadModel(modelPath, archPath = None):
    # LOAD THE MODEL
    #load serialized model
    if archPath is None:
        jsonFile = open(modelPath+'.json', 'r')
    else:
        jsonFile = open(archPath, 'r')

    sdfModel = tf.keras.models.model_from_json(jsonFile.read())
    jsonFile.close()
    #load weights
    sdfModel.load_weights(modelPath + '.h5')
    #sdfModel.summary()
    return sdfModel

if __name__ == "__main__":
    # this should handle folders of meshes, parallelizing the meshing to avail cores
    parser = argparse.ArgumentParser(description='given a sdf weight set, generate mesh')
    parser.add_argument('weightPath', help='path to weight sets!')
    parser.add_argument('meshPath', help='path to corresponding mesh geometries.') 
    parser.add_argument('--archPath', default=None)
    parser.add_argument('--res', default=128, type=int)
    args = parser.parse_args()

    trainedModels = list([f.split('.h5')[0] for f in os.listdir(args.weightPath) if '.h5' in f])

    cubeMarcher = gm.CubeMarcher()
    
    uniformGrid = cubeMarcher.createGrid(args.res)

    csvFile = open('results.csv', 'w')

    csvWriter = csv.writer(csvFile, delimiter=',')
    csvWriter.writerow(['Name', 'Grid Error', 'Surface Error', 'Importance Error'])

    for m in trainedModels:
        modelPath = os.path.join(args.weightPath, m)
        meshPath = os.path.join(args.meshPath, m)
        try:
            print("[INFO] Loading model: ", m)
            sdfModel = loadModel(modelPath, archPath=args.archPath)

            print("[INFO] Loading mesh: ", m)
            mesh = gm.Mesh(meshPath)

            print("[INFO] Inferring Grid")
            gridPred = sdfModel.predict(uniformGrid)
        
            print("[INFO] Inferring Surface Points")
            surfaceSampler = gm.PointSampler(mesh, ratio=0.0, std=0.0)
            surfacePts = surfaceSampler.sample(100000)
            surfacePred = sdfModel.predict(surfacePts)

            print("[INFO] Inferring Importance Points")
            impSampler = gm.PointSampler(mesh, ratio=0.1, std=0.01)
            impPts = impSampler.sample(100000)
            impPred = sdfModel.predict(impPts)

            print("[INFO] Calculating true sdf")
            sdf = gm.SDF(mesh)
            gridTrue = sdf.query(uniformGrid)
            impTrue = sdf.query(impPts)

            print("[INFO] Calculating Error")

            gridError = np.mean(np.abs(gridTrue - gridPred))
            surfaceError = np.mean(np.abs(surfacePred))
            impError = np.mean(np.abs(impTrue - impPred))

            print("[INFO] Grid Error: ", gridError)
            print("[INFO] Surface Error: ", surfaceError)   
            print("[INFO] Imp Error (loss): ", impError)

            csvWriter.writerow([m, gridError, surfaceError, impError])
            
        except Exception as e:
            print (e)
    
    csvFile.close()

    


