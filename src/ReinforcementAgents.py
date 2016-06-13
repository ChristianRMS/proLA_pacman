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


    def countFood(self, state, pacmanSpositionAfterMoving):
        food = state.getFood()
        nonEatableGhosts = self.getNonEatableGhosts(state)
        def stopCondition(curX, curY):
            return food[curX][curY] and not (curX, curY) in nonEatableGhosts

        return self.countSearch(state.getWalls(), pacmanSpositionAfterMoving, stopCondition)

    def getNearestFoodPosition(self,state, pacmanSpositionAfterMoving):
        food = state.getFood()
        nonEatableGhosts = self.getNonEatableGhosts(state)
        def stopCondition(curX,curY):
            return food[curX][curY] and not (curX, curY) in nonEatableGhosts
        return self.abstractBroadSearch(state.getWalls(), pacmanSpositionAfterMoving, stopCondition)[0]

    def getNextNonEatableGhost(self,state,pacmanSpositionAfterMoving):
        nonEatableGhosts = self.getNonEatableGhosts(state)
        def stopCondition(curX,curY):
            return (curX, curY) in nonEatableGhosts
        return self.abstractBroadSearch(state.getWalls(), pacmanSpositionAfterMoving, stopCondition)
    
    def ghostsFeature(self,state,nextNonEatableGhostArr, pacmanSpositionAfterMoving, direction, maximumPathLength, ghostSpeed):
        nearestIntersection = self.getNearestIntersectionInDirection(state, pacmanSpositionAfterMoving, direction)
        nextNonEatableGhost = nextNonEatableGhostArr[1]
        
        if nearestIntersection == None or nextNonEatableGhost == None:
            return None
    
        
        a = float(maximumPathLength)
        v = float(ghostSpeed)
        distanceBetweenPacmanAndGhost = self.abstractBroadResult(pacmanSpositionAfterMoving, nextNonEatableGhost)
        #bc = float(self.getDistanceBetweenGhostAndIntersection(state, nearestIntersection, nextNonEatableGhost))
        #dc = float(self.distanceToNearestIntersectionInDirection(state, pacmanSpositionAfterMoving, direction))
        bc = float(self.abstractBroadResult(nearestIntersection, nextNonEatableGhost))
        dc = float(self.abstractBroadResult(nearestIntersection, pacmanSpositionAfterMoving))
        if nextNonEatableGhost <= dc:
            bc = 0
        dcXv = float(dc * v)
        aADD = float(dcXv + a)
        bcSUB = float(aADD - bc)
        result = float(bcSUB/a)
        return result

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
                return dist #self.getNextNonEatableGhost(state,pos)[0] + dist
        return None
    
    def getDistanceBetweenGhostAndIntersection(self, state, intersection, ghostPosition):
        result = float('inf')
        if intersection != None and ghostPosition != None:
#           def stopCondition(curX,curY):
#               return (curX, curY) in intersection
         #   print "getDistanceBetweenGhostAndIntersection: " + str(intersection) + " | " + str(ghostPosition)
            result = self.abstractBroadResult(intersection, ghostPosition)
        if result == float('inf'):
            result = 0
        return result
    
    def distanceToNearestIntersectionInDirection(self, state, startposition, direction):    
            nearestIntersection = self.getNearestIntersectionInDirection(state, startposition, direction)
            
            if startposition != None and nearestIntersection != None:
              #  print "distanceToNearestIntersectionInDirection: " + str(startposition) + " | " + str(nearestIntersection)
                return self.abstractBroadResult(startposition, nearestIntersection)
            else:
                return 0
     
    def powerPillDuration(self, state):
        if self.powerPillCount == -1:
            self.powerPillCount = state.getCapsules()
        oldPowerPillCount = self.powerPillCount;
        self.powerPillCount = state.getCapsules()
        if(self.powerPillCount-oldPowerPillCount == 0):
            self.currentPowerPillDuration = self.powerPillDuration
            self.powerPillStartTime = time.time()
        else:
            self.powerPillCurrentTime = time.time
            self.currentPowerPillDuration = self.powerPillStartTime - self.powerPillCurrentTime + self.powerPillDuration
        if(self.currentPowerPillDuration > 0):
            return (self.powerPillDuration-self.currentPowerPillDuration)/self.powerPillDuration
        return 0
     
    def entrapmentFeature(self, nonEatableGhosts, state, pacmanPosition, direction): 
       # numberOfSafeJunctions = self.getNumberOfSafeIntersections(nonEatableGhosts, state, pacmanPosition, wayPoints)
       # numberOfSafeJunctionsInDirection = self.getNumberOfSafeIntersectionsInDircetion(nonEatableGhosts, state, pacmanPosition, wayPoints, direction)  
        ghosts = nonEatableGhosts
        saveIntersectionsCountInDir = 0
        saveIntersectionsCount = 0    
        #if intersections == None:
        #    return None 
        for intersection in self.intersections:
           # print "entrapmentFeature: " + str(pacmanPosition) + " | " + str(intersection)
            distanceToPacman = self.abstractBroadResult(pacmanPosition, intersection)
            nearestGhostDistance = float('inf')
            for ghost in ghosts:
            #    print "entrapmentFeature | GhostDis : " + str(ghost) + " | " + str(intersection)
                ghostDistance = self.abstractBroadResult(ghost, intersection)
                if ghostDistance < nearestGhostDistance:
                    nearestGhostDistance = ghostDistance
            if distanceToPacman < nearestGhostDistance:
                saveIntersectionsCount+=1  
                intersectionX, intersectionY = intersection
                pacmanPositionX, pacmanPositionY = pacmanPosition
                if direction == Directions.EAST:
                    if pacmanPositionX < intersectionX:
                       saveIntersectionsCountInDir+=1
                if direction == Directions.WEST:
                    if pacmanPositionX > intersectionX:
                       saveIntersectionsCountInDir+=1
                if direction == Directions.NORTH:
                    if pacmanPositionY < intersectionY:
                       saveIntersectionsCountInDir+=1     
                if direction == Directions.SOUTH:
                    if pacmanPositionY > intersectionY:
                       saveIntersectionsCountInDir+=1 
                                
        if saveIntersectionsCount == 0:
            return 0   
        #print("intersections: " + str(saveIntersectionsCount))
        #print("intersectionInDirection" + str(saveIntersectionsCountInDir))       
        return float(float((saveIntersectionsCount - saveIntersectionsCountInDir))/saveIntersectionsCount)
        
    
    def getNearestIntersectionInDirection(self, state, startposition, direction):
        result = None
        MinDistance = float('inf')
        
        for intersection in self.intersections: 
            intersectionX, intersectionY = intersection
            startX, startY = startposition
            if direction == Directions.EAST:
                if intersectionX <= startX and intersectionY != startY:
                       continue
            if direction == Directions.WEST:
               if intersectionX >= startX and intersectionY != startY:
                   continue
            if direction == Directions.NORTH:
               if intersectionY <= startY and intersectionX != startX:
                   continue
            if direction == Directions.SOUTH:
               if intersectionY >= startY and intersectionX != startX:
                   continue
        #    print "getNearestIntersectionInDirection: " + str(startposition) + " | " + str(intersection)   
            distance = self.abstractBroadResult(startposition, intersection)
            intersectionX, intersectionY = intersection
            startX, startY = startposition
            if direction == Directions.EAST:
                if intersectionX >= startX and intersectionY == startY:
                   if distance < MinDistance:
                       result = intersection 
                       minDistance = distance
            if direction == Directions.WEST:
               if intersectionX <= startX and intersectionY == startY:
                   if distance < MinDistance:
                       result = intersection
                       minDistance = distance
            if direction == Directions.NORTH:
               if intersectionY >= startY and intersectionX == startX:
                   if distance < MinDistance:
                       result = intersection
                       minDistance = distance
            if direction == Directions.SOUTH:
               if intersectionY <= startY and intersectionX == startX:
                   if distance < MinDistance:
                       result = intersection
                       minDistance = distance
        return result;
        
    def getStateSearch(self, state, direction, ghostSpeed = 0.8):
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
        searchResult['ghostFeature'] = self.ghostsFeature(state,nextNonEatableGhostArr, pacmanSpositionAfterMoving, direction, self.getMaximumDistance(state), ghostSpeed)
        searchResult['nearestFoodDist'] = self.getNearestFoodPosition(state,pacmanSpositionAfterMoving)
        searchResult['nearestEatableGhostDistances'] = self.getNextEatableGhost(state, pacmanSpositionAfterMoving)
        searchResult['entrapmentFeature'] = self.entrapmentFeature(nonEatableGhosts, state, pacmanSpositionAfterMoving, direction)
        searchResult['foodcount'] = self.countFood(state,pacmanSpositionAfterMoving)
        #searchResult['maximumDistance'] = self.getMaximumDistance(state)

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
    def getfeatures(self, state, direction, ghostSpeed = 0.8, lastAction = None, startFood = None):
        if self.hasMapCalc == False:
            self.hasMapCalc = True
            #print state.getWalls().__getitem__(1).__getitem__(1)
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
                                   #print " |x: " + str(x) + " |y: " +str(y) + " |xx: " + str(xx) + " |yy: " + str (yy) + " Value: " + str(self.dic[key])
            #print "Fertig!"
            self.createIntersections()
            pickle.dump( self.dic, open( "save.p", "wb" ) )
            
        features = myDict(0.0)
        #features['base'] = 1.0
        logging.debug("str " + str(state))
        logging.debug("dir " + str(direction))
        stateSearch = self.getStateSearch(state, direction, ghostSpeed)
        maxDistance = state.getWalls().width + state.getWalls().height #stateSearch['maxDistance'] #
        logging.debug("MaxDistance " + str(direction) + " " + str(maxDistance))
        if stateSearch['nearestFoodDist'] is not None:
            features['foodValuability'] = (float(float((maxDistance - stateSearch['nearestFoodDist'])) / maxDistance)) #/ maxDistance
            #print(str(direction) + " food " + str(stateSearch['nearestFoodDist']))
        if stateSearch['nearestGhostDistances'] is not None:
            features['ghostThreat'] = 1.0 - (float(maxDistance - stateSearch['nearestGhostDistances'])) / maxDistance
        if stateSearch['ghostFeature'] is not None:
            features['ghostFeature'] = 1.0-(float(stateSearch['ghostFeature'])) #/ maxDistance
#        else:
#            features['ghostThreat'] = float(maxDistance)
#        if stateSearch['nearestPowerPelletDist'] is not None:
#            logging.debug("PowerPelletDist " +  str(stateSearch['nearestPowerPelletDist']))
#            features['powerPelletValuability'] = (float(stateSearch['nearestPowerPelletDist'])) #/ maxDistance
#        else:
#            features['powerPelletValuability'] = 0.0
        if stateSearch['nearestEatableGhostDistances'] is not None:
            features['eatableGhosts'] = (float(float(maxDistance - stateSearch['nearestEatableGhostDistances'])/maxDistance)) #/ maxDistance
        if stateSearch['entrapmentFeature'] is not None:
            features["entrapment"] = 1.0 - float(stateSearch['entrapmentFeature'])
        if stateSearch['foodcount'] is not None:
            features['levelProgress'] = ((float(startFood)) - (float(stateSearch['foodcount']))) / (float(startFood))
        features['action'] = (float(1 if lastAction == direction else 0))
        features['maxDistance'] = maxDistance
        #features.divideAll(maxDistance)
        logging.debug(str(features))
        return features   
"""
class ReinforcementRAgent(game.Agent):
    def __init__(self, numTraining = 0):
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
        self.ghostSpeed = 0.8
        

    def safeListRemove(self,lst,item):
        try:
            lst.remove(item)
        except ValueError:
            pass

    def getCombinedValue(self,state, direction):
        combinedValue = 0.0
        features = self.ruleGenerator.getfeatures(state, direction)
        logging.debug("Features " + str(direction) + " " + str(features))
        for featureKey in features.keys():
            combinedValue += features[featureKey] * self.actionPower[featureKey]
        return combinedValue

    def updater(self,nextState):
        logging.debug("Start Updating")
        reward = self.calcReward(nextState)
        features = self.ruleGenerator.getfeatures(self.lastState, self.lastAction)
        combinatedValue = self.getCombinedValue(self.lastState, self.lastAction)
        maxPossibleFutureValue = self.getBestValue(nextState, self.legaldirections(nextState))
        for ruleKey in features.keys():
            difference = reward + self.gamma * maxPossibleFutureValue - combinatedValue
            logging.debug("Difference: " + str(difference))
            self.actionPower[ruleKey] = self.actionPower[ruleKey] + self.alpha * difference * features[ruleKey]
            #zur demo orginal QLearning
            #different = (reward + self.gamma * maxPossibleFutureValue - currentValue)
            #calcVal =  currentValue + self.alpha * different
        logging.debug("ActionPower: " + str(self.actionPower))
        #self.saving.setRatingForState(self.lastAction, self.lastState, calcVal)
        logging.debug("Stop Updating")

    def calcReward(self, state):
        return state.getScore() - self.lastState.getScore()

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

    def chooseAction(self, state):
        directions = self.legaldirections(state)
        logging.debug(str(directions))
        rnd = self.random.random()
        if self.epsilon > rnd:
            return self.random.choice(directions)
        else:
            return self.getBestDirection(self.lastState, directions)

    def legaldirections(self, state):
        directions = state.getLegalPacmanActions()
        self.safeListRemove(directions, Directions.LEFT)
        self.safeListRemove(directions, Directions.REVERSE)
        self.safeListRemove(directions, Directions.RIGHT)
        # self.safeListRemove(directions, Directions.STOP)
        return directions

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

    def final(self, state):
        self.updater(state)
        self.lastState = None
        self.lastAction = None
        #raw_input("Press Any Key ")
        if self.isInTraining():
            self.episodesSoFar += 1
            logging.info("Training " + str(self.episodesSoFar) + " of " + str (self.numTraining))
        else:
            self.epsilon = 0.0
            self.alpha = 0.0
            if state.isLose():
                #raw_input("Press Any Key ")
                pass

    def isInTraining(self):
        return self.episodesSoFar < self.numTraining

    def isInTesting(self):
        return not self.isInTraining()
"""

class NeuralAgent(game.Agent):
        
    def __init__(self, numTraining = 0):       
        self.network = NeuralController()
        self.network.printNetwork()
        #if os.path.isfile("netSave.p"):
        #    self.network.printNetwork()
        #    self.network.load()
        #    print "loaded"
        #    self.network.printNetwork()
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
        self.ghostSpeed = 0.8
        self.lastlastAction = None
        self.startFood = 0
        self.features = None

    def safeListRemove(self,lst,item):
        try:
            lst.remove(item)
        except ValueError:
            pass

    def getCombinedValue(self, state, direction):
        if self.startFood == 0:
            self.startFood = self.ruleGenerator.getStateSearch(state, direction)['foodcount']
        self.features = self.ruleGenerator.getfeatures(state, direction, self.ghostSpeed, self.lastAction, self.startFood)
        shortestPillDistance = self.features['foodValuability']
        shortestGhostDistance = self.features['ghostThreat']
        actionFeature = self.features['action']
        entrapment = self.features["entrapment"]
        eatableGhost = self.features['eatableGhosts']
        ghost = self.features['ghostFeature']
        levelProgress = self.features['levelProgress']
        action = self.network.calculateAction(shortestPillDistance,ghost,eatableGhost)
        #action = self.network.calculateAction(shortestPillDistance,shortestGhostDistance, eatableGhost, actionFeature, entrapment)
        #print(str(direction) + " " + str(shortestGhostDistance))
        #print(str(direction) + " " + str(shortestPillDistance))
        #print(str(direction) + " entrapment: " + str(entrapment))
        #print(str(direction) + " ghost" + str(ghost))
        
        return action

    def updater(self,nextState):
        reward = self.calcReward(nextState)
        combinatedValue = self.getCombinedValue(self.lastState, self.lastAction)
        maxPossibleFutureValue = self.getBestValue(nextState, self.legaldirections(nextState))
        ds = SupervisedDataSet(3,1)
        shortestPillDistance = self.features['foodValuability']
        shortestGhostDistance = self.features['ghostThreat']
        actionFeature = self.features['action']
        entrapment = self.features["entrapment"]
        eatableGhost = self.features['eatableGhosts']
        ghost = self.features['ghostFeature']
        levelProgress = self.features['levelProgress']
        self.network.logNetwork(shortestPillDistance, ghost,eatableGhost,reward)
        ds.addSample((shortestPillDistance, ghost,eatableGhost), (reward + self.gamma * maxPossibleFutureValue - combinatedValue))
        self.network.getTrainer().trainOnDataset(ds)
        
    def calcReward(self, state):
        return state.getScore() - self.lastState.getScore()

    def getAction(self, state):
        #self.network.printNetwork()
        self.lastlastAction = self.lastAction
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

    def observationFunction(self, state, intersections):
        self.wayPoints = intersections
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
            #self.network.printNetwork()

    def isInTraining(self):
        return self.episodesSoFar < self.numTraining

    def isInTesting(self):
        return not self.isInTraining()