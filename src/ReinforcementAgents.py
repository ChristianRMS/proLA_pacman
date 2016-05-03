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
        #print "startX" + str(startX)
        #print "startY" + str(startY)
        #print "stop" + str(stopCondition)
        
        
        openList = [(startX, startY, 0)]
        closedList = set()
        while openList:
            curX, curY, dist = openList.pop(0)
            #print "curX" + str(curX)
            #print "curY" + str(curY)
            if not (curX, curY) in closedList:
                closedList.add((curX, curY))
                if stopCondition(curX, curY):
                    return [dist,(curX, curY)]
                for (sucX, sucY) in self.getMovableDirections(curX, curY,field):
                    openList.append((sucX, sucY, dist + 1))
        return [None,None]

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
        return self.abstractBroadSearch(state.getWalls(), pacmanSpositionAfterMoving, stopCondition)[0]
    
    def ghostsFeature(self,state,pacmanSpositionAfterMoving, direction, intersections, maximumPathLength, ghostSpeed):
        nearestIntersection = self.getNearestIntersectionInDirection(state, pacmanSpositionAfterMoving, direction, intersections)
        
        def stopCondition(curX,curY):
            return (curX, curY) in nonEatableGhosts
        
        a = maximumPathLength
        print "path" + str(a)
        v = ghostSpeed
        print "v" + str(v)
        bc = self.getDistanceBetweenGhostAndIntersection(state, nearestIntersection, stopCondition)
        print "bc" + str(bc)
        dc = self.distanceToNearestIntersectionInDirection(state, pacmanSpositionAfterMoving, direction, intersections)
        print "dc" + str(dc)
        dcXv = float(dc * v)
        aADD = float(a + dcXv)
        bcSUB = float(aADD - bc)
        result = float(bcSUB/a)
        return result

    def getNextEatableGhost(self, state, pacmanSpositionAfterMoving):
        powerPellets = state.getCapsules()
        nonEatableGhosts = self.getNonEatableGhosts(state)
        eatableGhosts = self.getEatableGhosts(state)
        field = state.getWalls()
        for ghost in nonEatableGhosts:
            #print ghost
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
                return self.getNextNonEatableGhost(state,pos) + dist
        return None
    
    def getDistanceBetweenGhostAndIntersection(self, state, intersection, ghostPosition):
        result = float('inf')
        #print "ghostPosition" + str(ghostPosition)
        #print "inter " + str(intersection)
        if intersection != None and ghostPosition != None:
#            def stopCondition(curX,curY):
#                return (curX, curY) in intersection
            result = self.abstractBroadSearch(state.getWalls(), ghostPosition, ghostPosition)
        if result == float('inf'):
            result = 0
        return result
    
    def distanceToNearestIntersectionInDirection(self, state, startposition, direction, intersections):    
            nearestIntersection = self.getNearestIntersectionInDirection(state, startposition, direction, intersections)
            
            if startposition != None and nearestIntersection != None:
                def stopCondition(curX,curY):
                        return (curX, curY) in nearestIntersection
                return self.abstractBroadSearch(state.getWalls(), startposition, stopCondition)
            else:
                return 0
    def getNearestIntersectionInDirection(self, state, startposition, direction, intersections):
        result = None
        MinDistance = float('inf')
        print "intersections" + str(intersections)
        if intersections != None:
            for intersection in intersections:   
                def stopCondition(curX,curY):
                    print "curX" + str(curX)
                    print "curY" + str(curY)
                    print "intersection" + str(intersection)
                    print str((curX, curY) in intersection)
                    curX = float(curX)
                    curY = float(curY)
                    return (curX, curY) in intersection
                #print "intersection" + str(intersection)
                distance = self.abstractBroadSearch(state.getWalls(), startposition, stopCondition)
                print "distance" + str(distance)
                intersectionX, intersectionY = intersection
                startX, startY = startposition
                if direction == Directions.EAST:
                   if intersectionX >= startX:
                       if distance < MinDistance:
                           result = intersection 
                           minDistance = distance
                if direction == Directions.WEST:
                   if intersectionX <= startX:
                       if distance < MinDistance:
                           result = intersection
                           minDistance = distance
                if direction == Directions.NORTH:
                   if intersectionY >= startY:
                       if distance < MinDistance:
                           result = intersection
                           minDistance = distance
                if direction == Directions.SOUTH:
                   if intersectionY <= startY:
                       if distance < MinDistance:
                           result = intersection
                           minDistance = distance
        return result;
        
    def getStateSearch(self, state, direction, intersections = None, ghostSpeed = 0.8):
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
        searchResult['nearestEatableGhostDistances'] = self.getNextEatableGhost(state, pacmanSpositionAfterMoving)
        searchResult['nearestGhostDistances'] = self.getNextNonEatableGhost(state, pacmanSpositionAfterMoving)
        searchResult['ghostFeature'] = self.ghostsFeature(state,pacmanSpositionAfterMoving, direction, intersections, self.getMaximumDistance(state), ghostSpeed)
        searchResult['nearestFoodDist'] = self.getNearestFoodPosition(state,pacmanSpositionAfterMoving)
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
    def getfeatures(self, state, direction, intersections = None, ghostSpeed = 0.8, lastAction = None):
        features = myDict(0.0)
        #features['base'] = 1.0
        logging.debug("str " + str(state))
        logging.debug("dir " + str(direction))
        stateSearch = self.getStateSearch(state, direction, intersections, ghostSpeed)
        maxDistance = state.getWalls().width + state.getWalls().height #stateSearch['maxDistance'] #
        logging.debug("MaxDistance " + str(direction) + " " + str(maxDistance))
        if stateSearch['nearestFoodDist'] is not None:
            logging.debug("FoodDist " +  str(stateSearch['nearestFoodDist']))
            features['foodValuability'] = (float(stateSearch['nearestFoodDist'])) #/ maxDistance
        if stateSearch['nearestGhostDistances'] is not None:
            logging.debug("ghostThreat " +  str(stateSearch['nearestGhostDistances']))
            features['ghostThreat'] = (float(stateSearch['nearestGhostDistances'])) #/ maxDistance
        if stateSearch['ghostFeature'] is not None:
            features['ghostFeature'] = (float(stateSearch['ghostFeature'])) #/ maxDistance
            print "ghost" + str(features['ghostFeature'])
#        else:
#            features['ghostThreat'] = float(maxDistance)
#        if stateSearch['nearestPowerPelletDist'] is not None:
#            logging.debug("PowerPelletDist " +  str(stateSearch['nearestPowerPelletDist']))
#            features['powerPelletValuability'] = (float(stateSearch['nearestPowerPelletDist'])) #/ maxDistance
#        else:
#            features['powerPelletValuability'] = 0.0
        if stateSearch['nearestEatableGhostDistances'] is not None:
            features['eatableGhosts'] = (float(float(maxDistance - stateSearch['nearestEatableGhostDistances'])/maxDistance)) #/ maxDistance
            print "eatable" + str(features['eatableGhosts'])
            print str(maxDistance)
            print str(stateSearch['nearestEatableGhostDistances'])
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
        self.intersections = []
        self.ghostSpeed = 0.8
        self.lastlastAction = None

    def safeListRemove(self,lst,item):
        try:
            lst.remove(item)
        except ValueError:
            pass

    def getCombinedValue(self, state, direction):
        features = self.ruleGenerator.getfeatures(state, direction, self.intersections, self.ghostSpeed, self.lastAction)
        shortestPillDistance = features['foodValuability']
        shortestGhostDistance = features['ghostThreat']
        #print shortestGhostDistance
        #print shortestPillDistance
        #shortestEatableGhost = features['eatableGhosts']
        """print "StepVA" + str(self.step)
        #print "spd: " + str(shortestPillDistance)
        #print "sgd: " + str(shortestGhostDistance)"""
        action = self.network.calculateAction(shortestPillDistance,shortestGhostDistance)
        return action

    def updater(self,nextState):
        reward = self.calcReward(nextState)
        features = self.ruleGenerator.getfeatures(self.lastState, self.lastAction, self.intersections, self.ghostSpeed, self.lastlastAction)
        combinatedValue = self.getCombinedValue(self.lastState, self.lastAction)
        maxPossibleFutureValue = self.getBestValue(nextState, self.legaldirections(nextState))
        ds = SupervisedDataSet(2,1)
        shortestPillDistance = features['foodValuability']
        shortestGhostDistance = features['ghostThreat']
        #shortestEatableGhost = features['eatableGhosts']
        """print "StepUP" + str(self.step)
        #print "spd: " + str(shortestPillDistance)
        #print "sgd: " + str(shortestGhostDistance)"""
        ds.addSample((shortestPillDistance, shortestGhostDistance), (reward + self.gamma * maxPossibleFutureValue - combinatedValue))
        self.network.getTrainer().trainOnDataset(ds)
        #for ruleKey in features.keys():
        #    difference = reward + self.gamma * maxPossibleFutureValue - combinatedValue
        #    logging.debug("Difference: " + str(difference))
        #    self.actionPower[ruleKey] = self.actionPower[ruleKey] + self.alpha * difference * features[ruleKey]
        #    #zur demo orginal QLearning
        #    different = (reward + self.gamma * maxPossibleFutureValue - currentValue)
        #    calcVal =  currentValue + self.alpha * different
        #logging.debug("ActionPower: " + str(self.actionPower))
        #self.saving.setRatingForState(self.lastAction, self.lastState, calcVal)
        #logging.debug("Stop Updating")
        #self.step = self.step + 1

    def calcReward(self, state):
        return state.getScore() - self.lastState.getScore()

    def getAction(self, state):
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

    def observationFunction(self, state, intersections):
        self.intersections = intersections
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
            self.network.printNetwork()
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