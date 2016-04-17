from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers.trainer import Trainer
from pybrain.supervised.trainers.backprop import BackpropTrainer
class NeuralController:
   
      
    #--Konstruktor--
    def __init__(self):
        self.net = buildNetwork(2,50,1);
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
    
    def calculateAction(self, ghostDistance, pillDistance):
        result = self.net.activate([ghostDistance, pillDistance])
        return result[0]
    
    def getTrainer(self):
        return self.trainer
    
    