from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers.trainer import Trainer
class KNNAgent:
    #"""Die Klasse f�r das Neuronale Netzwerk"""
      
    #"""--Konstruktor--"""
    def __init__(self):
        self.net = buildNetwork(2,50,1);
        self.trainer = Trainer(self.net)
        self.data = []
    
    #"""Feature Eins - LevelProgress"""
    def levelProgress(amountOfPillsTotal, amountOfPillsRemain):
        return (amountOfPillsTotal - amountOfPillsRemain) / amountOfPillsTotal
    #"""Feature Zwei - PowerPillen""" 
    def powerPill(totalDurationOfPill, timeSinceConsumed):
        return (totalDurationOfPill - timeSinceConsumed) / totalDurationOfPill
    #"""Feature Drei - Pillen - F�r jede Richtung!(Wenn ein SingleAction Netzwerk"""
    def pillen(maximumPathLength, shortestDistanceInDirection):
        return (maximumPathLength - maximumPathLength) / maximumPathLength
    #"""Feature Vier - Geister die angreifen. F�r jede Richtung!(Wenn ein SingleAction Netzwerk"""
    def ghosts(maximumPathLength, ghostSpeed, distGhostInterInDirection, distPacmanInterInDirection):
        return (maximumPathLength + (distPacmanInterInDirection * ghostSpeed) - (distGhostInterInDirection / maximumPathLength))
    #"""Feature F�nf - Scared Ghosts. F�r jede Richtung!(Wenn ein SingleAction Netzwerk"""
    def scaredGhost(maximumPathLength, shortestDistanceScaredGhostInDirection):    
        return (maximumPathLength - shortestDistanceScaredGhostInDirection) / maximumPathLength
    #"Feature Sechs - Entrapment. F�r jede Richtung!(Wenn ein SingleAction Netzwerk"
    def entrapment(totalAmountOfSafeRoutes, amountOfSafeRoutesInDirection):
        return (totalAmountOfSafeRoutes - amountOfSafeRoutesInDirection) / totalAmountOfSafeRoutes
    #"Feature Sieben - Action. F�r Jede Richtung! Glaube ich."
    def action(currentDirction):
        return currentDirction
    
    #"Ausf�hren des Netzwerks"
    def calculateAction(ghostDistance, pillDistance):
        result = net.activate([ghostDistance, pillDistance])
        print result[0] 
        return result
    
    def getTrainer(self):
        return self.trainer
    
    