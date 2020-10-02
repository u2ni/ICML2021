
NO_LIBIGL = False

import model

if not NO_LIBIGL:
  import geometry as gm

from sdfsequencer import SDFSequence

import numpy as np
import tensorflow as tf

import os
import argparse

from tqdm import tqdm

import h5py

def createSequences(sdf, grid, pointSampler, batchSize, epochLength=10**6, reuseEpoch=True, useSphericalCoordinates=False):
  if reuseEpoch:
    # We just precompute one epoch and reuse each time!
    queryPts = pointSampler.sample(epochLength)
    S = sdf.query(queryPts)

    trainData = np.concatenate((queryPts,S), axis = 1)

    trainSDF = SDFSequence(
      trainData,
      None,
      batchSize
    )
  else:
    # continuous sampling sequence 
    trainSDF = SDFSequence(
      sdf,
      pointSampler,
      batchSize,
      epochLength
    )

  if grid is None:
    evalSDF = None
  else:
    gridS = sdf.query(grid)
    validationData = np.concatenate((grid,gridS), axis = 1)

    # fixed grid sequence
    evalSDF = SDFSequence(
      validationData, 
      None,
      batchSize
    )

  return trainSDF, evalSDF


def singleModelTrain(
  meshFn, 
  precomputedFn,
  config,
  showVis = True):

  outputDir = os.path.abspath(config.saveDir)

  if (not meshFn is None):
    cubeMarcher = gm.CubeMarcher()    
    mesh = gm.Mesh(meshFn, doNormalize=True)

    samplingMethod = config.samplingMethod

    sdf = gm.SDF(mesh)

    if samplingMethod['type'] == 'SurfaceUniform':
      pointSampler = gm.PointSampler(mesh, ratio = samplingMethod['ratio'], std = samplingMethod['std'])
    elif samplingMethod['type'] == 'Uniform':
      pointSampler = gm.PointSampler(mesh, ratio = 1.0)
    elif samplingMethod['type'] == 'Importance':
      pointSampler = gm.ImportanceSampler(mesh, int(config.epochLength/samplingMethod['ratio']), samplingMethod['weight'])
    else:
      raise("uhhhh")

    # create data sequences
    validationGrid = cubeMarcher.createGrid(config.validationRes) if config.validationRes > 0 else None
    sdfTrain, sdfEval = createSequences(sdf, validationGrid, pointSampler, config.batchSize, config.epochLength)

  elif (not precomputedFn is None) :
    # precomputed!
    if 'h5' in precomputedFn:
      if config.queryPath is None:
        raise("Must supply path to queries if using h5 data!")
      else:
        f = h5py.File(config.queryPath, 'r')
        queries = np.array(f['queries'])
        f = h5py.File(precomputedFn, 'r')
        S = np.array(f['sdf'])
        S = S.reshape((S.shape[0],1))

        if config.samplingMethod['type'] == 'Importance':
          importanceSampler = gm.ImportanceSampler(None, S.shape[0], config.samplingMethod['weight'])
          queries,S = importanceSampler.sampleU(int(S.shape[0]/config.samplingMethod['ratio']), queries, S)

        precomputedData = {
          'train': np.concatenate((queries, S), axis=1)
        }
    else:
      precomputedData = np.load(precomputedFn)

    trainData = precomputedData['train']
    validationData = precomputedData['validation'] if 'validation' in precomputedData else None

    sdfTrain = SDFSequence(
      trainData,
      None,
      config.batchSize
    )

    if validationData is None:
      sdfEval = None
    else:
      sdfEval = SDFSequence(
        validationData,
        None,
        config.batchSize
      )

  else:
    raise(ValueError("uhh I need data"))


  # create model
  sdfModel = model.SDFModel(config)

  # train the model
  sdfModel.train(
    trainGenerator = sdfTrain,
    validationGenerator = sdfEval,
    epochs = config.epochs
  )

  if showVis:
    # predict against grid
    rGrid = cubeMarcher.createGrid(config.reconstructionRes)
    S = sdfModel.predict(rGrid)

    # plot results
    sdfModel.plotTrainResults()

    cubeMarcher.march(rGrid,S)
    marchedMesh = cubeMarcher.getMesh() 
    marchedMesh.show()

  if (not (outputDir == None)):
    sdfModel.save()
    if showVis:
      marchedMesh.save(os.path.join(outputDir,config.name + '.obj'))
      sdfModel.plotTrainResults(show = False, save = True)

def parseArgs():
  parser = argparse.ArgumentParser(description='Train model to predict sdf of a given mesh, by default visualizes reconstructed mesh to you, and plots loss.')
  # data specific
  parser.add_argument('inputPath', help='path to input mesh/ folder of meshes', default=None)
  parser.add_argument('--queryPath', help='path to h5 dataset containining queries', default=None)
  parser.add_argument('--reuseEpoch',type=int, default=True,help='option to reuse first epoch, or regenerate every time!')
  parser.add_argument('--outputDir', help='directory to save model and mesh artifacts', default='../results/')
  parser.add_argument('--validationRes', type=int, help='resolution of validation grid', default= 32)
  parser.add_argument('--reconstructionRes', type=int, help='resolution of output sdf grid', default= 0)
  parser.add_argument('--showVis', type=int, default=False, help='0 to disable vis for headless')
  parser.add_argument('--epochs', type=int, help='epochs to run training job(s) for', default= 100)
  parser.add_argument('--epochLengthPow', type=int, default=6, help='10**epochLengthPow')
  parser.add_argument('--learningRate', type=float, default= 0.001, help='starting lr for training')
  parser.add_argument('--loss', type=str, default='l1')
  parser.add_argument('--batchSize', type=int, default=2048,help='batch size for training')
  parser.add_argument('--activation', default='relu', type=str)
  parser.add_argument('--firstLayerHiddenSize', type=int, default=32)
  parser.add_argument('--numLayers', type=int, default=8)
  parser.add_argument('--samplingMethod', type=str, default='Importance')
  parser.add_argument('--importanceWeight', type=int, default=60)
  parser.add_argument('--suffix', type=str, default='')
  parser.add_argument('--gpu', type=int, default=0)
  parser.add_argument('--writeOutEpochs', type=int, default=0)
  return parser.parse_args()

def isMesh(fn):
  if fn.split('.')[-1] in ['off','stl', 'obj']:
    return True
  else:
    return False
      
if __name__ == "__main__":
  args = parseArgs()

  tfConfig = tf.compat.v1.ConfigProto()
  os.environ["CUDA_VISIBLE_DEVICES"]="{}".format(args.gpu)  
  tfConfig.gpu_options.allow_growth = True
  sess = tf.compat.v1.Session(config=tfConfig)
  tf.compat.v1.keras.backend.set_session(sess)  

  assert(os.path.exists(args.inputPath))
  inputPath = args.inputPath

  if os.path.isfile(inputPath):
    files = [inputPath]
  else:
    supportedFormats = ['off','stl', 'obj', 'npz', 'h5']
    files = [i for i in os.listdir(inputPath)]
    files = [i for i in files if i.split('.')[-1] in supportedFormats]
    files = [os.path.join(inputPath, i) for i in files]

  # default config for all experiments (specific experiments may override!)
  config = model.Config()
  config.saveDir = args.outputDir
  config.batchSize = args.batchSize
  config.learningRate = args.learningRate
  config.epochLength = 10**args.epochLengthPow
  config.epochs = args.epochs
  config.validationRes = args.validationRes
  config.reconstructionRes = args.reconstructionRes
  config.activation = args.activation
  config.numLayers = args.numLayers
  config.hiddenSize = args.firstLayerHiddenSize
  config.lossType = args.loss
  config.queryPath = args.queryPath
  config.saveWeightsEveryEpoch = args.writeOutEpochs

  if (args.samplingMethod == 'Importance'):
    print("Importance Sampling!")
    config.samplingMethod = {
      'weight': args.importanceWeight,
      'ratio': 0.1,
      'type': 'Importance'
    }
  elif (args.samplingMethod == 'Surface'):
    config.samplingMethod = {
      'std': 0.01,
      'ratio': 0.1,
      'type': 'SurfaceUniform'
    }
  elif (args.samplingMethod == 'Uniform'):
    config.samplingMethod = {
      'type': 'Uniform'
    }
  else:
    print("INVALID SAMPLING METHOD EXITING")
    exit()
    
  for dataFile in files:
    config.name = os.path.splitext(os.path.basename(dataFile))[0] + args.suffix

    # train model on single mesh given
    singleModelTrain(
      meshFn = dataFile if isMesh(dataFile) else None,
      precomputedFn = dataFile if not isMesh(dataFile) else None, 
      config = config,
      showVis = args.showVis
    )