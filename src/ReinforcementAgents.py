from bfsSearch import ReinforcementSearch
from game import Directions
import game
import logging
from neuralController import NeuralController
from pybrain.datasets.supervised import SupervisedDataSet
import random
from networkx.algorithms.operators.binary import intersection
from boto.dynamodb.condition import NULL
from dask.array.ghost import nearest
from Cython.Shadow import typeof
import pickle
import os.path 
from bokeh.core.enums import Direction
from bokeh.util.logconfig import level
import time
from sqlalchemy.sql.expression import intersect

class AbstractQState():
    def __init__(self, state, direction):
        #self.state = state
        # TODO: check this => these keys are not in featue but in stateSearch/ searchResult
        features = RuleGenerator().getStateSearch(state,direction)
        self.ghostThreat = features['nearestGhostDistances']
        self.foodDistance = features['nearestFoodDist']
        self.powerPelletDist = features['nearestPowerPelletDist']
        # self.eatableGhosts = features['nearestEatableGhostDistances']
        #self.direction = direction

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            # return self.ghostThreat == other.ghostThreat and self.foodDistance == other.foodDistance and self.powerPelletDist == other.powerPelletDist and self.eatableGhosts == other.eatableGhosts
            return self.ghostThreat == other.ghostThreat and self.foodDistance == other.foodDistance and self.powerPelletDist == other.powerPelletDist
        else:
            return False
    def __hash__(self):
        # return hash(hash(self.ghostThreat) + hash(self.foodDistance) + hash(self.powerPelletDist) + hash(self.eatableGhosts))
        return hash(hash(self.ghostThreat) + hash(self.foodDistance) + hash(self.powerPelletDist))

class Saving():
    def __init__(self, evalFn="scoreEvaluation"):
        self.savedStates = {}

    def getRatingForNextState(self, direction, state):
        abstractState = AbstractQState(state, direction)
        value = self.savedStates.get(abstractState)
        if value == None:
            return 0
        else:
            return value

    def setRatingForState(self, direction, state, value):
        abstractState = AbstractQState(state, direction)
        self.savedStates[abstractState] = value

    def getBestDirection(self, state, directions):
        bestVal = float('-inf')
        bestDirection = None
        for direction in directions:
            tmpValue = self.getRatingForNextState(direction, state)
            if bestVal < tmpValue:
                bestVal = tmpValue
                bestDirection = direction
        return bestDirection

    def getBestValue(self, state, directions):
        bestDirection = self.getBestDirection(state,directions)
        if bestDirection:
            return self.getRatingForNextState(bestDirection, state)
        else:
            return 0.0

    def __repr__(self):
        return str(self.savedStates)

class ReinforcementQAgent(game.Agent):
    def __init__(self, numTraining = 0):
         self.saving = Saving()
         self.random = random.Random()
         self.lastState = None
         self.lastAction = None
         self.alpha = 0.1
         self.gamma = 0.5
         self.epsilon = 0.2
         self.numTraining = int(numTraining)
         self.episodesSoFar = 0

    def getAction(self, state):
        logging.debug(str(state))
        self.lastAction = self.chooseAction(state)

        return self.lastAction

    def chooseAction(self, state):
        directions = self.legaldirections(state)
        rnd = self.random.random()
        if self.epsilon > rnd:
            logging.debug("random " + str(rnd) + " gamma " + str(self.epsilon))
            return self.random.choice(directions)
        else:
            return self.saving.getBestDirection(self.lastState, directions)

    def calcReward(self, state):
        return state.getScore() - self.lastState.getScore()

    def legaldirections(self, state):
        directions = state.getLegalPacmanActions()
        self.safeListRemove(directions, Directions.LEFT)
        self.safeListRemove(directions, Directions.REVERSE)
        self.safeListRemove(directions, Directions.RIGHT)
        #self.safeListRemove(directions, Directions.STOP)
        return directions

    def safeListRemove(self,lst,item):
        try:
            lst.remove(item)
        except ValueError:
            pass

    def updater(self, state):
        reward = self.calcReward(state)
        currentValue = self.saving.getRatingForNextState(self.lastAction, self.lastState)
        maxPossibleFutureValue = self.saving.getBestValue(state, self.legaldirections(state))
        calcVal =  currentValue + self.alpha * (reward + self.gamma * maxPossibleFutureValue - currentValue)
        self.saving.setRatingForState(self.lastAction, self.lastState, calcVal)

    def observationFunction(self, state):
        if self.lastState:
            self.updater(state)
        self.lastState = state
        return state

    def final(self, state):
        self.updater(state)
        self.lastState = None
        self.lastAction = None
        if self.isInTraining():
            self.episodesSoFar += 1
            logging.info("Training " + str(self.episodesSoFar) + " of " + str(self.numTraining))
        else:
            self.epsilon = 0.0
            self.alpha = 0.0

    def isInTraining(self):
        return self.episodesSoFar < self.numTraining

    def isInTesting(self):
        return not self.isInTraining()

class myDict(dict):
    def __init__(self, default):
        self.default = default

    def __getitem__(self, key):
        self.setdefault(key, self.default)
        return dict.__getitem__(self, key)

    def sumAll(self):
        sumAllret = 0.0
        for key in self.keys():
            sumAllret += self[key]
        return sumAllret

    def normalize(self):
        sumAlla = self.sumAll()
        if sumAlla == 0.0:
            newValue = float(1)/len(self)
            for key in self.keys():
                self[key] = newValue
        else:
            for key in self.keys():
                self[key] = self[key] / sumAlla

    def divideAll(self, value):
        for key in self.keys():
            self[key] = float(self[key]) / value

class RuleGenerator():
    
    def __init__(self):
        self.hasMapCalc = False
        self.powerPillCount =-1
        self.powerPillDuration = 60
        self.powerPillStartTime = 0
        self.powerPillCurrentTime = 0
        self.currentPowerPillDuration = 0
        self.dic = dict()
        self.wayPoints = []
        self.intersections = []
        self.ghostSpeed = 0.8
        self.startFood = 0

    def directionToCoordinate(self, direction):
        if direction == Directions.NORTH:
            return (0,1)
        elif direction == Directions.SOUTH:
            return (0,-1)
        elif direction == Directions.EAST:
            return (1,0)
        elif direction == Directions.WEST:
            return (-1,0)
        else:
            return (0,0)
    
    def coodinateToDirection(self, coordinate):
        if coordinate == (0,1):
            return Directions.NORTH
        elif coordinate == (0,-1):
            return Directions.SOUTH
        elif coordinate == (1,0):
            return Directions.EAST
        elif coordinate == (-1,0):
            return Directions.WEST
        else:
            return None
    def getMovableDirections(self,posX,posY, walls):
        lst = [(0,0)]
        try:
            if not walls[posX][posY+1]:
                lst.append((posX, posY+1))
        except IndexError:
            pass

        try:
            if not walls[posX][posY-1]:
                lst.append((posX, posY-1))
        except IndexError:
            pass

        try:
            if not walls[posX-1][posY]:
                lst.append((posX-1, posY))
        except IndexError:
            pass

        try:
            if not walls[posX+1][posY]:
                lst.append((posX+1, posY))
        except IndexError:
            pass

        return lst
    
    def getMovableDirectionList(self,posX,posY, walls):
        lst = []
        try:
            if not walls[posX][posY+1]:
                lst.append(Directions.NORTH)
        except IndexError:
            pass

        try:
            if not walls[posX][posY-1]:
                lst.append(Directions.SOUTH)
        except IndexError:
            pass

        try:
            if not walls[posX-1][posY]:
                lst.append(Directions.WEST)
        except IndexError:
            pass

        try:
            if not walls[posX+1][posY]:
                lst.append(Directions.EAST)
        except IndexError:
            pass

        return lst

    '''
    Benutzt eine Breitensuche, um die Entfernung zwischen 2 koordinaten zu ermitteln
    Liefert ein Array zurueck, bei dem das Element mit dem index 0 die entfernung zwischen der startingPosition
    und der stopCondition ist und das Element mit dem Index 1 ist die Koordinate dem die stopCondition wahr wurde, z.B. die Positionen der Geister
    '''
    def abstractBroadSearch(self,field, startingPosition, stopCondition):
        startX, startY = startingPosition
        counter = 0
        
        
        openList = [(startX, startY, 0)]
        closedList = set()
        while openList:
            counter+=1
            curX, curY, dist = openList.pop(0)
            curX = int(curX)
            curY = int(curY)
            if not (curX, curY) in closedList:
                closedList.add((curX, curY))
                if stopCondition(curX, curY):
                    return [dist,(curX, curY)]
                for (sucX, sucY) in self.getMovableDirections(curX, curY,field):
                    openList.append((sucX, sucY, dist + 1))
        return [None,None]
    
    '''
    Liefert die Entfernung zwischen den uebergebenen Koordinaten.
    Liest die Entfernung aus einer Liste mit den Entfernungen aller koordinaten-kombinationen aus.
    Die Liste wird einmal zu beginn des Spiels erstellt.
    '''
    def abstractBroadResult(self, startingPosition, stopPosition):
        x,y = startingPosition
        x = int(x)
        y = int(y)       
        xx,yy = stopPosition
        xx = int(xx)
        yy = int(yy)
        key = str(x) + str(y) + str(xx) + str (yy)
     #   print key
        if key not in self.dic:
            return None
        return self.dic[key]
    
    '''
    Erstellt eine Liste mit allen Kreuzungen des Spielfelds und speichert diese in der Klassenvariable "intersections".
    '''
    def createIntersections(self):
        self.intersections = []
        for pos in self.wayPoints:
            x,y = pos
            posLeft = (x-1,y)
            posRight = (x+1,y)
            posUp = (x, y+1)
            posDown = (x, y-1)
            
            resLeftUp = self.abstractBroadResult(posLeft, posUp) == 2
            resRightUp = self.abstractBroadResult(posRight, posUp) == 2
            resLeftDown = self.abstractBroadResult(posLeft, posDown) == 2
            resRightDown = self.abstractBroadResult(posRight, posDown) == 2
            
            if resLeftDown or resLeftUp or resRightDown or resRightUp:
                self.intersections.append(pos)
    
    '''
    Zaehlt wie haeufig die stopCondition true wurde und gibt die Anzahl zurueck.
    Benutzt eine Breitensuche
    '''
    def countSearch(self, field, startingPosition, stopCondition):
        startX, startY = startingPosition
        openList = [(startX, startY, 0)]
        closedList = set()
        count = 0
        while openList:
            curX, curY, dist = openList.pop(0)
            if not (curX, curY) in closedList:
                closedList.add((curX, curY))
                if stopCondition(curX, curY):
                    count = count+1
                for (sucX, sucY) in self.getMovableDirections(curX, curY, field):
                    openList.append((sucX, sucY, dist + 1))
        return count

    '''
    Gibt die Anzahl der essbaren Pillen auf dem Spielfeld zurueck
    '''
    def countFood(self, state, pacmanSpositionAfterMoving):
        food = state.getFood()
        nonEatableGhosts = self.getNonEatableGhosts(state)
        def stopCondition(curX, curY):
            return food[curX][curY] and not (curX, curY) in nonEatableGhosts

        return self.countSearch(state.getWalls(), pacmanSpositionAfterMoving, stopCondition)

    '''
    Liefert die entfernung der naechsten essbaren pille zurueck
    '''
    def getNearestFoodDistance(self,state, pacmanSpositionAfterMoving):
        food = state.getFood()
        nonEatableGhosts = self.getNonEatableGhosts(state)
        def stopCondition(curX,curY):
            return food[curX][curY] and not (curX, curY) in nonEatableGhosts
        return self.abstractBroadSearch(state.getWalls(), pacmanSpositionAfterMoving, stopCondition)[0]

    '''
    Liefert ein Array mit der Entfernung zu dem naechsten nicht-essbaren Geist und dessen koordinate zurueck
    '''
    def getNextNonEatableGhost(self,state,pacmanSpositionAfterMoving):
        nonEatableGhosts = self.getNonEatableGhosts(state)
        def stopCondition(curX,curY):
            return (curX, curY) in nonEatableGhosts
        return self.abstractBroadSearch(state.getWalls(), pacmanSpositionAfterMoving, stopCondition)
    
    

    '''
    Liefert die Entfernung zum naechsten essbaren Geist oder falls kein essbarer Geist vohanden ist,
    die Entfernung zur naechsten Power-Pille
    '''
    def getNextEatableGhost(self, state, pacmanSpositionAfterMoving):
        powerPellets = state.getCapsules()
        nonEatableGhosts = self.getNonEatableGhosts(state)
        eatableGhosts = self.getEatableGhosts(state)
        field = state.getWalls()
        for ghost in nonEatableGhosts:
            x,y = ghost
            field[int(x)][int(y)]=False
        if len(eatableGhosts) != 0:
            def stopConditionEatableGhost(curX,curY):
                return (curX, curY) in eatableGhosts
            return self.abstractBroadSearch(field, pacmanSpositionAfterMoving, stopConditionEatableGhost)[0]
        elif len(powerPellets) != 0:
            def stopConditionPallet(curX,curY):
                if (curX, curY) in powerPellets and not (curX, curY) in nonEatableGhosts:
                    return True
                else:
                    return False
            pallet = self.abstractBroadSearch(field, pacmanSpositionAfterMoving, stopConditionPallet)
            dist = pallet[0]
            pos = pallet[1]
            if pos:
                return dist # self.getNextNonEatableGhost(state,pos)[0] + dist
        return None                  
    
   
    
    '''
    Liefert die in der uebergebenen Richtung naechste Kreuzung.
    '''
    def getNearestIntersectionInDirection(self, state, startposition, direction):
        result = None
        MinDistance = float('inf')
        
        for intersection in self.intersections: 
            intersectionX, intersectionY = intersection
            startX, startY = startposition 
            
            east = direction == Directions.EAST and intersectionX > startX
            west = direction == Directions.WEST and intersectionX < startX
            south = direction == Directions.SOUTH and intersectionY < startX
            north = direction == Directions.NORTH and intersectionY > startX
                
            if east or west or south or north:
                distance = self.abstractBroadResult(startposition, intersection)
                if distance < MinDistance:
                    MinDistance = distance
                    result = intersection
        return result;
    
    '''
    GhostFeature aus dem MS.Pacman Paper
    Vergleicht wie lange Pacman zu der Kreuzung in der uebergebenen Richtung braucht, und wie lang der Geist zu der Kreuzung braucht
    '''
    def ghostsFeature(self,state,nextNonEatableGhostArr, pacmanSpositionAfterMoving, direction):
        nearestIntersection = self.getNearestIntersectionInDirection(state, pacmanSpositionAfterMoving, direction)
        nextNonEatableGhost = nextNonEatableGhostArr[1]
        
        if nearestIntersection == None or nextNonEatableGhost == None:
            return None
        
        maximumPathLength = self.getMaximumDistance(state)
        distanceGhostToIntersection = float(self.abstractBroadResult(nearestIntersection, nextNonEatableGhost))
        distancePacmanToIntersection = float(self.abstractBroadResult(nearestIntersection, pacmanSpositionAfterMoving))
        result = float(float(distancePacmanToIntersection * self.ghostSpeed - distanceGhostToIntersection) / maximumPathLength)
        return result
    
    '''
    LevelProgressFeature aus dem Ms. Pacman Paper
    Zeigt an, wieviel Prozent der Pillen schon gefressen wurden
    '''
    def levelProgressFeature(self, state, stateSearch):
        return (float(self.startFood - stateSearch['foodcount']) / float(self.startFood))
    
    '''
    Action Feature aus dem Ms. Pacman Paper
    Gibt eine 1 zurueck, wenn Pacman die Richtung fuer die der Featurewert berechnet wird, die selbe ist wie die in der Pacman sich als
    letztes bewegt hat, ansonsten 0
    '''
    def actionFeature(self, lastAction, direction):
        return 1 if lastAction == direction else 0
    
    #TODO
    def PowerPillFeature(self): 
       return None
    
    #TODO
    def entrapmentFeature(self): 
       return None
        
    def getStateSearch(self, state, direction):
        vecX, vecY = self.directionToCoordinate(direction)
        posX, posY = state.getPacmanPosition()
        pacmanSpositionAfterMoving = (posX + vecX, posY + vecY)
        #food = state.getFood()
        #walls = state.getWalls()
        # eatableGhosts = self.getEatableGhosts(state)
        #nonEatableGhosts = self.getNonEatableGhosts(state)
        #powerPellets = state.getCapsules()
        #openList = [(posX + vecX, posY + vecY, 0)]
        #closedList = set()
        searchResult = myDict(None)
        #maxDistance = -1
        #searchResult['nearestFoodDist'] = None
        #searchResult['nearestPowerPelletDist'] = None
        #logging.debug("powerPellets = " + str(powerPellets))
        #while openList:
            #curX, curY, dist = openList.pop(0)
            #if not (curX, curY) in closedList:
                #closedList.add((curX, curY))
                #if (curX, curY) in nonEatableGhosts:
                #    logging.debug("###################################### nonEatableGhosts= " + str(nonEatableGhosts))
                #    logging.debug("###################################### state.getGhostStates = " + str(not state.getGhostStates()[0].isScared()))
                #    if not searchResult.has_key('nearestGhostDistances'):
                #        searchResult['nearestGhostDistances'] = dist
                #if (searchResult['nearestPowerPelletDist'] is None) and (curX, curY) in powerPellets and not (curX, curY) in nonEatableGhosts:
                #    searchResult['nearestPowerPelletDist'] = dist
                #    searchResult['nearestPowerPelletPos'] = (curX, curY)
                # if (curX, curY) in eatableGhosts:
                #      logging.debug("###################################### eatableGhosts= " + str(eatableGhosts))
                #      logging.debug("###################################### state.getGhostStates = " + str(state.getGhostStates()[0].isScared()))
                #     if not searchResult.has_key('nearestEatableGhostDistances'):
                #         searchResult['nearestEatableGhostDistances'] = dist
                #for (sucX, sucY) in self.getMovableDirections(curX, curY,walls):
                #    openList.append((sucX, sucY, dist + 1))
                #maxDistance = max(maxDistance, dist)
        nextNonEatableGhostArr = self.getNextNonEatableGhost(state, pacmanSpositionAfterMoving)
        nonEatableGhosts = self.getNonEatableGhosts(state)
        searchResult['nearestGhostDistances'] = nextNonEatableGhostArr[0]
        searchResult['ghostFeature'] = self.ghostsFeature(state,nextNonEatableGhostArr, pacmanSpositionAfterMoving, direction)
        searchResult['nearestFoodDist'] = self.getNearestFoodDistance(state,pacmanSpositionAfterMoving)
        searchResult['nearestEatableGhostDistances'] = self.getNextEatableGhost(state, pacmanSpositionAfterMoving)
        searchResult['entrapmentFeature'] = self.entrapmentFeature()
        searchResult['foodcount'] = self.countFood(state, pacmanSpositionAfterMoving)

        return searchResult

    maxDistance = None
    def getMaximumDistance(self, state):
        if (RuleGenerator.maxDistance == None):
            RuleGenerator.maxDistance = (ReinforcementSearch(state)).getMaximumDistance()
        return RuleGenerator.maxDistance

    def getEatableGhosts(self, state):
        eatableGhosts = []
        ghostStates = state.getGhostStates()
        for ghostState in ghostStates:
            if ghostState.isScared():
                eatableGhosts.append(ghostState.getPosition())
        logging.debug("############################################# eatableGhosts = " + str(eatableGhosts))
        return eatableGhosts

    def getNonEatableGhosts(self, state):
        nonEatableGhosts = []
        ghostStates = state.getGhostStates()
        for ghostState in ghostStates:
            if not ghostState.isScared():
                nonEatableGhosts.append(ghostState.getPosition())
        logging.debug("############################################# nonEatableGhosts = " + str(nonEatableGhosts))
        return nonEatableGhosts

    # TODO: insert features here
    def getfeatures(self, state, direction, lastAction = None):
        if self.startFood == 0:
            self.startFood = self.getStateSearch(state, direction)['foodcount']
        if self.hasMapCalc == False:
            self.hasMapCalc = True
            for x in range(0, state.getWalls().width):
                for y in range(0, state.getWalls().height):
                    for xx in range(0, state.getWalls().width):
                        for yy in range (0, state.getWalls().height):
                           key = str(x) + str(y) + str(xx) + str (yy)
                           def stopCondition(curX,curY):
                               result = curX == xx and curY == yy
                               return result
                           startPosition = (x, y)
                           field = state.getWalls()
                           if not state.getWalls().__getitem__(x).__getitem__(y):
                               if not state.getWalls().__getitem__(xx).__getitem__(yy):
                                   self.dic[key] = self.abstractBroadSearch(field, startPosition, stopCondition)[0]
                                   if startPosition not in self.wayPoints:
                                       self.wayPoints.append(startPosition)
            self.createIntersections()
            pickle.dump( self.dic, open( "save.p", "wb" ) )
            
        features = myDict(0.0)
        logging.debug("str " + str(state))
        logging.debug("dir " + str(direction))
        stateSearch = self.getStateSearch(state, direction)
        #maxDistance = state.getWalls().width + state.getWalls().height #stateSearch['maxDistance']
        maxDistance = self.getMaximumDistance(state)
        logging.debug("MaxDistance " + str(direction) + " " + str(maxDistance))
        if stateSearch['nearestFoodDist'] is not None:
            features['foodValuability'] = (float(stateSearch['nearestFoodDist']) / maxDistance) #/ maxDistance
        
        #Distanz zum naechsten Geist
        if stateSearch['nearestGhostDistances'] is not None:
            features['ghostThreat'] = (float(stateSearch['nearestGhostDistances'])) / maxDistance
        
        #Ghost Feature aus dem Ms. Pacman Paper
        if stateSearch['ghostFeature'] is not None:
            features['ghostFeature'] = float(stateSearch['ghostFeature'])
        
        if stateSearch['nearestEatableGhostDistances'] is not None:
            features['eatableGhosts'] = (float(stateSearch['nearestEatableGhostDistances']/maxDistance))
        if stateSearch['entrapmentFeature'] is not None:
            features["entrapment"] = float(stateSearch['entrapmentFeature'])
        if stateSearch['foodcount'] is not None:
            features['levelProgress'] = self.levelProgressFeature(state, stateSearch)
        features['action'] = self.actionFeature(lastAction, direction)
        features['maxDistance'] = maxDistance
        logging.debug(str(features))
        return features   

class NeuralAgent(game.Agent):
        
    def __init__(self, numTraining = 0):       
        self.network = NeuralController()
        self.network.printNetwork()
        #if os.path.isfile("netSave.p"):
        #    self.network.load()
        self.actionPower = myDict(0.0)
        self.ruleGenerator = RuleGenerator()
        self.random = random.Random()
        self.lastState = None
        self.lastAction = None
        self.alpha = 0.5
        self.gamma = 0.5
        self.epsilon = 0.1
        self.numTraining = int(numTraining)
        self.episodesSoFar = 0
        self.step = 0
        self.features = None

    def safeListRemove(self,lst,item):
        try:
            lst.remove(item)
        except ValueError:
            pass

    '''
    Liefert die Summe aller Featurewerte fuer die uebergebene Richtung
    '''
    def getCombinedValue(self, state, direction):
        
        self.features = self.ruleGenerator.getfeatures(state, direction, self.lastAction)
        shortestPillDistance = self.features['foodValuability']
        shortestGhostDistance = self.features['ghostThreat']
        actionFeature = self.features['action']
        entrapment = self.features["entrapment"]
        eatableGhost = self.features['eatableGhosts']
        ghost = self.features['ghostFeature']
        levelProgress = self.features['levelProgress']
        action = self.network.calculateAction(ghost, shortestPillDistance, eatableGhost, levelProgress)      
        return action

    '''
    Diese Methode trainiert das Neuronale Netz.
    Dazu werden erst die Featurewerte und der Reward fuer die Richtung in die Pacman gegangen ist errechnet.
    Danach wird das Netzwerk durch den trainer mit hilfe der Featurewerte und des Rewards trainiert.
    '''
    def updater(self,nextState):
        if self.isInTraining():
            reward = self.calcReward(nextState)
            combinatedValue = self.getCombinedValue(self.lastState, self.lastAction)
            maxPossibleFutureValue = self.getBestValue(nextState, self.legaldirections(nextState))
            ds = SupervisedDataSet(4,1)
            shortestPillDistance = self.features['foodValuability']
            shortestGhostDistance = self.features['ghostThreat']
            actionFeature = self.features['action']
            entrapment = self.features["entrapment"]
            eatableGhost = self.features['eatableGhosts']
            ghost = self.features['ghostFeature']
            levelProgress = self.features['levelProgress']
            self.network.logNetwork(ghost, shortestPillDistance, eatableGhost, levelProgress,reward)
            ds.addSample((ghost, shortestPillDistance, eatableGhost, levelProgress), (reward + self.gamma * maxPossibleFutureValue - combinatedValue))
            self.network.getTrainer().trainOnDataset(ds) #das Netzwerk mit den featurewerten und dem Ergebnis trainieren
        
    '''
    Berechnet den Reward fuer die letzte aktion
    '''
    def calcReward(self, state):
        return state.getScore() - self.lastState.getScore()

    '''
    Wird von game.py aufgerufen und liefert dem Spiel die naechste Aktion, die Pacman ausfuehren soll
    '''
    def getAction(self, state):
        logging.debug("Start GetAction")
        self.lastAction = self.chooseAction(state)
        logging.debug("Action Power: " + str(self.actionPower))
        if self.isInTesting():
#            raw_input("Press Any Key ")
            pass
        logging.debug("Chosen Action: " + str(self.lastAction))
        logging.debug("Stop GetAction")
        logging.debug(str(self.lastAction))
        return self.lastAction

    '''
    Waehlt aus, ob eine Zufaellige Aktion oder die vom Netzwerk am besten bewertete Aktion gewaehlt werden soll und gibt diese zurueck
    '''
    def chooseAction(self, state):
        directions = self.legaldirections(state)
        logging.debug(str(directions))
        rnd = self.random.random()
        if self.epsilon > rnd:
            if self.isInTraining():
                return self.random.choice(directions)
        return self.getBestDirection(self.lastState, directions)
    
    def legaldirections(self, state):
        directions = state.getLegalPacmanActions()
        self.safeListRemove(directions, Directions.LEFT)
        self.safeListRemove(directions, Directions.REVERSE)
        self.safeListRemove(directions, Directions.RIGHT)
        self.safeListRemove(directions, Directions.STOP)
        return directions

    '''
    Liefert die vom Netzwerk am besten bewertete Aktion
    '''
    def getBestDirection(self, state, directions):
        bestVal = float('-inf')
        bestDirection = None
        logging.debug("Possible Directions" + str(directions))
        for direction in directions:
            tmpValue = self.getCombinedValue(state, direction)
            logging.debug("Combinated Value " + str(direction) + " " + str(tmpValue))
            logging.debug(str(tmpValue))
            if bestVal < tmpValue:
                bestVal = tmpValue
                bestDirection = direction
        return bestDirection

    def getBestValue(self, state, directions):
        bestDirection = self.getBestDirection(state,directions)
        if bestDirection:
            return self.getCombinedValue(state, bestDirection)
        else:
            return 0.0
    
    '''
    Wird nach jeder Aktion von game.py aufgerufen
    '''
    def observationFunction(self, state):
        if self.lastState:
            self.updater(state)
        else:
            if not self.isInTraining():
               self.epsilon = 0.0
               self.alpha = 0.0
               pass
        self.lastState = state
        #raw_input("Press Any Key ")
        return state

    '''
    Wird am Ende jedes Spiels von game.py aufgerufen
    '''
    def final(self, state):
        self.updater(state)
        self.lastState = None
        self.lastAction = None
        #raw_input("Press Any Key ")
        self.network.printNetwork()
        if self.isInTraining():
            self.episodesSoFar += 1
            logging.info("Training " + str(self.episodesSoFar) + " of " + str (self.numTraining))
            print("Training " + str(self.episodesSoFar) + " of " + str (self.numTraining))
        else:
            self.epsilon = 0.0
            self.alpha = 0.0
            if state.isLose():
                #raw_input("Press Any Key ")
                pass
            self.network.save()

    '''
    Gibt zurueck, ob sich pacman noch in der Trainingsphase befindet
    '''
    def isInTraining(self):
        return self.episodesSoFar < self.numTraining
    
    '''
    Gibt zurueck, ob sich pacman noch in der Testphase befindet
    '''
    def isInTesting(self):
        return not self.isInTraining()