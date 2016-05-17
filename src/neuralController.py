from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers.trainer import Trainer
from pybrain.supervised.trainers.backprop import BackpropTrainer
from pybrain.structure.networks.feedforward import FeedForwardNetwork
from pybrain.structure.modules.linearlayer import LinearLayer
from pybrain.structure.modules.sigmoidlayer import SigmoidLayer
from pybrain.structure.connections.full import FullConnection
from pybrain.tools.shortcuts import buildNetwork
import pickle

class NeuralController:
       
    #--Konstruktor--
    def __init__(self):
        self.net = FeedForwardNetwork()
        
        self.inLayer = LinearLayer(5)
        self.hiddenLayer = SigmoidLayer(50)
        self.outLayer = LinearLayer(1)
        
        self.net.addInputModule(self.inLayer)
        self.net.addModule(self.hiddenLayer)
        self.net.addOutputModule(self.outLayer)
        
        self.in_to_hidden = FullConnection(self.inLayer, self.hiddenLayer)
        self.hidden_to_out = FullConnection(self.hiddenLayer, self.outLayer)
        
        self.net.addConnection(self.in_to_hidden)
        self.net.addConnection(self.hidden_to_out)
        
        self.net.sortModules()
        
        self.trainer = BackpropTrainer(self.net)
        self.data = []
        
        
    
    #Feature Eins - LevelProgress
    def levelProgress(self, amountOfPillsTotal, amountOfPillsRemain):
        return (amountOfPillsTotal - amountOfPillsRemain) / amountOfPillsTotal
    #Feature Zwei - PowerPillen 
    def powerPill(self, totalDurationOfPill, timeSinceConsumed):
        return (totalDurationOfPill - timeSinceConsumed) / totalDurationOfPill
    #Feature Drei - Pillen - Fuer jede Richtung!(Wenn ein SingleAction Netzwerk
    def pillen(self, maximumPathLength, shortestDistanceInDirection):
        return (maximumPathLength - maximumPathLength) / maximumPathLength
    #Feature Vier - Geister die angreifen. Fuer jede Richtung!(Wenn ein SingleAction Netzwerk
    def ghosts(self, maximumPathLength, ghostSpeed, distGhostInterInDirection, distPacmanInterInDirection):
        return (maximumPathLength + (distPacmanInterInDirection * ghostSpeed) - (distGhostInterInDirection / maximumPathLength))
    #Feature Fuenf - Scared Ghosts. Fuer jede Richtung!(Wenn ein SingleAction Netzwerk
    def scaredGhost(self, maximumPathLength, shortestDistanceScaredGhostInDirection):    
        return (maximumPathLength - shortestDistanceScaredGhostInDirection) / maximumPathLength
    #Feature Sechs - Entrapment. Fuer jede Richtung!(Wenn ein SingleAction Netzwerk
    def entrapment(self, totalAmountOfSafeRoutes, amountOfSafeRoutesInDirection):
        return (totalAmountOfSafeRoutes - amountOfSafeRoutesInDirection) / totalAmountOfSafeRoutes

    def action(self, currentDirction):
        return currentDirction
    
    def calculateAction(self, entrapment, shortestPillDistance,shortestGhostDistance, eatableGhost, ghost):
        result = self.net.activate([entrapment, shortestPillDistance,shortestGhostDistance, eatableGhost, ghost])
        return result[0]
    
    def printNetwork(self):
        print "Training abgeschlossen"
        print "Input to Hidden Connections:"
        print self.in_to_hidden.params
        print "Hidden to Output Connections:"
        print self.hidden_to_out.params
    
    def getTrainer(self):
        return self.trainer
    
    def save(self):
        fileObject = open('netSave', 'w')
        pickle.dump(self.net, fileObject)
        fileObject.close()
    
    def load(self):
        fileObject = open('netSave', 'r')
        self.net = pickle.load(fileObject)
        fileObject.close()
