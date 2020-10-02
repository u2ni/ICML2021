import matplotlib.pyplot as plt
import numpy as np
 
class LearningRateDecay:
	def plot(self, epochs, title="Learning Rate Schedule"):
		# compute the set of learning rates for each corresponding
		# epoch
		lrs = [self(i) for i in epochs]
 
		# the learning rate schedule
		plt.style.use("ggplot")
		plt.figure()
		plt.plot(epochs, lrs)
		plt.title(title)
		plt.xlabel("Epoch #")
		plt.ylabel("Learning Rate")
		plt.show()

class StepDecay(LearningRateDecay):
	def __init__(self, initLR=0.01, factor=0.25, dropEvery=10):
		# store the base initial learning rate, drop factor, and
		# epochs to drop every
		self.initLR = initLR
		self.factor = factor
		self.dropEvery = dropEvery
 
	def __call__(self, epoch):
		# compute the learning rate for the current epoch
		exp = np.floor((1 + epoch) / self.dropEvery)
		LR = self.initLR * (self.factor ** exp)
 
		# return the learning rate
		return float(LR)

class PolynomialDecay(LearningRateDecay):
	def __init__(self, maxEpochs=100, initLR=0.01, power=1.0):
		# store the maximum number of epochs, base learning rate,
		# and power of the polynomial
		self.maxEpochs = maxEpochs
		self.initLR = initLR
		self.power = power
 
	def __call__(self, epoch):
		# compute the new learning rate based on polynomial decay
		decay = (1 - (epoch / float(self.maxEpochs))) ** self.power
		LR = self.initLR * decay
 
		# return the new learning rate
		return float(LR)

if __name__ == "__main__":
	import argparse
	parser = argparse.ArgumentParser(description='Learning rate schedules visualizer')
	parser.add_argument('--schedule', type=int, help= 'step: 0, polynomial: 1')
	parser.add_argument('--lr', default=0.001,type=float, help='initial learning rate')
	parser.add_argument('--factor', default=0.1,type=float, help='step size for step schedule')
	parser.add_argument('--dropEpochs', default=15,type=int, help='how many epochs til step drop')
	parser.add_argument('--power', default=1,type=int, help='polynomial decay exponent')
	parser.add_argument('--epochs', default=100, type=int,help='number of epochs to decay over')

	args = parser.parse_args()

	if args.schedule == 0:
		s = StepDecay(initLR=args.lr, factor=args.factor, dropEvery=args.dropEpochs)
	elif args.schedule == 1:
		s = PolynomialDecay(maxEpochs=args.epochs, initLR=args.lr,power=args.power)

	s.plot(range(args.epochs))
	
