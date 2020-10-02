#!/usr/bin/env python

import tensorflow as tf 
import numpy as np
import matplotlib.pyplot as plt
import os
import schedules

from tensorboard.plugins.hparams import api as hp

#tf.enable_v2_behavior()
tf.compat.v1.disable_eager_execution()

class Config(object):
    """Holds model hyperparams and data information.

    The config class is used to store various hyperparameters and dataset
    information parameters. Model objects are passed a Config() object at
    instantiation.
    """
    hiddenSize = 32
    batchSize = 2048
    numLayers = 8
    activation = 'elu'
    workers = 4
    saveDir = ''
    name = 'sdfModel'
    lossType = 'l1' # l1,l2,clamp,min
    clampValue = 0.1
    optimizer = 'adam'

    #loggin options (for review !)
    logTensorboard = True
    logHparam = False
    useMultiProcessing = False

    #learning rate specific params
    learningRate = 0.0005

    #frame param
    useFrames = False
    saveWeightsEveryEpoch = False

class SDFModel:
  optimizer = None
  model = None
  loss = None
  history = None

  def __init__(self, config):
    self.config = config
    self.createLoss()
    self.createOpt()
    self.build()

  def build(self):
    tf.keras.backend.clear_session()

    inputs = tf.keras.Input(shape= (3,))

    x = tf.keras.layers.Dense(
      self.config.hiddenSize,
      input_shape=(3,),
      activation = self.config.activation
    )(inputs)

    for _ in range(self.config.numLayers - 1):
        x = tf.keras.layers.Dense(
          self.config.hiddenSize,
          activation = self.config.activation
        )(x)
    
    outputs = tf.keras.layers.Dense(
        1,
        activation='tanh',
    )(x)

    self.model = tf.keras.Model(inputs=inputs, outputs=outputs)
    self.model.compile(
      optimizer = self.optimizer,
      loss = self.loss,
      metrics = self.modelMetrics()
    )
    self.model.summary()

  def modelMetrics(self):
    def max_pred(labels, predictions):
      return tf.keras.backend.max(predictions)
    def min_pred(labels, predictions):
      return tf.keras.backend.min(predictions)
    def avg_pred(labels, predictions):
      return tf.keras.backend.mean(predictions) 
    def mse(labels, predictions):
      return tf.keras.metrics.mean_squared_error(labels,predictions)
    def mae(labels, predictions):
      return tf.keras.metrics.mean_absolute_error(labels,predictions)
    def mape(labels, predictions):
      return tf.keras.metrics.mean_absolute_percentage_error(labels,predictions)
    def overshot(labels, predictions):
      return tf.abs(tf.math.minimum(labels - predictions, 0.0), name="overshotLoss")
    def inOut(labels,predictions):
      return 0.5*tf.math.maximum(0.0, 1 - tf.math.sign(labels) * tf.math.sign(predictions))

    metrics = [mse,mae]#,mape,overshot, inOut]

    return (metrics)

  def getModelWeights(self):
    allWeights = []
    for l in self.model.layers:
      w = l.get_weights()
      allWeights.append(w)
    return allWeights

  def tensorboardCallback(self):
    boardPath = os.path.join( os.path.join(self.config.saveDir,'logs'), self.config.name)
    return tf.keras.callbacks.TensorBoard(
          log_dir=boardPath, 
          histogram_freq=0,  
          profile_batch=0,
          write_graph=True)

  def saveEveryEpochCallback(self):
    fn = os.path.join(os.path.join(self.config.saveDir,'checkpoints'), self.config.name)
    fn = fn+'_weights.{epoch:03d}.h5'
    return tf.keras.callbacks.ModelCheckpoint(
        fn, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=True, period=1)

  def hparamCallback(self):
    hparams = {
      'learningRate' : self.config.learningRate,
      'hiddenSize': self.config.hiddenSize,
      'numberOfLayers': self.config.numLayers,
      'batchSize': self.config.batchSize,
      'numberParams': self.model.count_params()
    }

    boardPath = os.path.join( os.path.join(self.config.saveDir,'logs'), self.config.name)
    return hp.KerasCallback(boardPath, hparams)
  
  """Train model"""
  def train(self, trainGenerator, validationGenerator, epochs, schedule=None):
    callbacks = []
    
    if self.config.logTensorboard:
      callbacks.append(self.tensorboardCallback())
    if self.config.logHparam: 
      callbacks.append(self.hparamCallback())
    if self.config.saveWeightsEveryEpoch:
      callbacks.append(self.saveEveryEpochCallback())

    # LR decay on plateau
    rlrop = tf.keras.callbacks.ReduceLROnPlateau(
      monitor='loss', 
      factor=0.8, 
      patience=30,
      min_lr=0.000001,
      verbose=1 #tell me when you reduce!
    )
    callbacks.append(rlrop)

    if validationGenerator is None:
      self.history = self.model.fit(
        x = trainGenerator,
        steps_per_epoch = len(trainGenerator),
        epochs = epochs, 
        shuffle = False,
        use_multiprocessing=False,
        workers=self.config.workers,
        max_queue_size=1000,
        callbacks=callbacks
      )
    else:
      self.history = self.model.fit(
        x = trainGenerator,
        validation_data = validationGenerator, 
        steps_per_epoch = len(trainGenerator),
        validation_steps = len(validationGenerator),
        epochs = epochs, 
        shuffle = False,
        use_multiprocessing=False,
        workers=self.config.workers,
        max_queue_size=1000,
        callbacks=callbacks
      )

  def plotTrainResults(self, show=True, save=False):
    legend = ['Train']
    
    plt.plot(self.history.history['loss'])

    if 'val_loss' in self.history.history:
      plt.plot(self.history.history['val_loss'])
      legend.append('Val')

    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')

    plt.legend(legend, loc='upper left')

    if save:
      plt.savefig(os.path.join(self.config.saveDir,self.config.name + '.png'))
    if show:
      plt.show()


  def save(self):
    #serialize model to json
    modelJson = self.model.to_json()
    with open(os.path.join(self.config.saveDir,self.config.name + '.json'), 'w') as jsonFile:
      jsonFile.write(modelJson)
    #save weights
    self.model.save_weights(os.path.join(self.config.saveDir,self.config.name + '.h5'))
  
  def load(self, modelFolder = None):
    if modelFolder == None:
      modelFolder = self.config.saveDir

    #load serialized model
    jsonFile = open(os.path.join(modelFolder,self.config.name + '.json'), 'r')
    self.model = tf.keras.models.model_from_json(jsonFile.read())
    jsonFile.close()
    #load weights
    self.model.load_weights(os.path.join(modelFolder,self.config.name + '.h5'))

  def predict(self, data):
    return self.model.predict(data, batch_size = self.config.batchSize, verbose=1)

  def _clampLoss(self,yTrue, yPred):
    return tf.keras.losses.mean_absolute_error(
      tf.clip_by_value(yTrue, -self.config.clampValue, self.config.clampValue),
      tf.clip_by_value(yPred, -self.config.clampValue, self.config.clampValue)
    )

  def _minLoss(self, yTrue, yPred):
    surfaceDistanceMult = 50

    maeLoss = tf.abs(yTrue - yPred) 
    #overshotLoss = tf.abs(tf.math.minimum(yTrue - yPred, 0.0), name="overshotLoss")
    # hinge loss
    #insideOutLoss = 0.5*tf.math.maximum(0.0, 1 - tf.math.sign(yTrue) * tf.math.sign(yPred))
    surfaceWeight = tf.math.exp(-surfaceDistanceMult*tf.math.abs(yTrue))

    return tf.reduce_mean(
      (
        maeLoss
        #+ overshotWeight*overshotLoss
        #+ insideOutWeight*insideOutLoss    
      )*surfaceWeight
    )

  def _weightedL1(self, yTrue, yPred):
    #hypothesis: we care less about things far from surface.
    return tf.reduce_mean(tf.abs(yTrue-yPred)*tf.math.exp(-50*tf.math.abs(yTrue)))

  def createLoss(self):
    if self.config.lossType ==  'l1':
      self.loss = tf.keras.losses.MeanAbsoluteError()
    elif self.config.lossType == 'l2':
      self.loss = tf.keras.losses.MeanSquaredError()
    elif self.config.lossType == 'clamp':
      self.loss = self._clampLoss
    elif self.config.lossType == 'min':
      self.loss = self._minLoss
    elif self.config.lossType == 'weighted':
      self.loss = self._weightedL1
    else:
      raise ('INVALID LOSS TYPE')

  def createOpt(self):
    self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.config.learningRate)



