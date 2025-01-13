# Report on `multiAgents.py`

## Overview
The `multiAgents.py` file is part of the Pacman AI projects developed at UC Berkeley. It contains implementations of various agents that play the Pacman game using different strategies. The file includes reflex agents, minimax agents, alpha-beta pruning agents, and expectimax agents.

## ReflexAgent
The `ReflexAgent` class inherits from the `Agent` class and chooses actions based on a state evaluation function.

- **getAction**: This method selects the best action based on the evaluation function.
- **evaluationFunction**: This method evaluates the game state by considering the distance to the closest food and the distance to the closest ghost.

```python
class ReflexAgent(Agent):
    def getAction(self, gameState):
        legalMoves = gameState.getLegalActions()
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)
        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        foodDistances = [manhattanDistance(newPos, food) for food in newFood.asList()]
        closestFoodDistance = min(foodDistances) if foodDistances else 0

        ghostDistances = [manhattanDistance(newPos, ghostState.getPosition()) for ghostState in newGhostStates]
        closestGhostDistance = min(ghostDistances) if ghostDistances else 0

        score = successorGameState.getScore()
        score -= closestFoodDistance
        if closestGhostDistance > 0:
            score += max(closestGhostDistance, 5)
        return score

## MultiAgentSearchAgent
The `MultiAgentSearchAgent` class is an abstract class that provides common elements for multi-agent searchers like Minimax, AlphaBeta, and Expectimax agents.

```python
class MultiAgentSearchAgent(Agent):
    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)
```

## MinimaxAgent
The `MinimaxAgent` class implements the minimax algorithm to choose the best action for Pacman.

- **getAction**: Returns the minimax action from the current game state.
- **minimax**: Recursively calculates the best score for an agent using the minimax algorithm.

```python
class MinimaxAgent(MultiAgentSearchAgent):
    def getAction(self, gameState):
        action, score = self.minimax(0, 0, gameState)
        return action

    def minimax(self, curr_depth, agent_index, gameState):
        if agent_index >= gameState.getNumAgents():
            agent_index = 0
            curr_depth += 1
        if curr_depth == self.depth:
            return None, self.evaluationFunction(gameState)

        best_score, best_action = None, None
        if agent_index == 0:
            for action in gameState.getLegalActions(agent_index):
                next_game_state = gameState.generateSuccessor(agent_index, action)
                _, score = self.minimax(curr_depth, agent_index + 1, next_game_state)
                if best_score is None or score > best_score:
                    best_score = score
                    best_action = action
        else:
            for action in gameState.getLegalActions(agent_index):
                next_game_state = gameState.generateSuccessor(agent_index, action)
                _, score = self.minimax(curr_depth, agent_index + 1, next_game_state)
                if best_score is None or score < best_score:
                    best_score = score
                    best_action = action
        if best_score is None:
            return None, self.evaluationFunction(gameState)
        return best_action, best_score
```

## AlphaBetaAgent
The `AlphaBetaAgent` class implements the minimax algorithm with alpha-beta pruning to optimize the search process.

- **getAction**: Returns the minimax action using alpha-beta pruning.
- **alphaBeta**: Recursively calculates the best score for an agent using alpha-beta pruning.

```python
class AlphaBetaAgent(MultiAgentSearchAgent):
    def getAction(self, gameState):
        def alphaBeta(curr_depth, agent_index, gameState, alpha, beta):
            if agent_index >= gameState.getNumAgents():
                agent_index = 0
                curr_depth += 1
            if curr_depth == self.depth:
                return self.evaluationFunction(gameState)
            if agent_index == 0:
                value = float('-inf')
                for action in gameState.getLegalActions(agent_index):
                    value = max(value, alphaBeta(curr_depth, agent_index + 1, gameState.generateSuccessor(agent_index, action), alpha, beta))
                    if value > beta:
                        return value
                    alpha = max(alpha, value)
                return value
            else:
                value = float('inf')
                for action in gameState.getLegalActions(agent_index):
                    value = min(value, alphaBeta(curr_depth, agent_index + 1, gameState.generateSuccessor(agent_index, action), alpha, beta))
                    if value < alpha:
                        return value
                    beta = min(beta, value)
                return value

        best_action = None
        alpha = float('-inf')
        beta = float('inf')
        value = float('-inf')
        for action in gameState.getLegalActions(0):
            new_value = alphaBeta(0, 1, gameState.generateSuccessor(0, action), alpha, beta)
            if new_value > value:
                value = new_value
                best_action = action
            alpha = max(alpha, value)
        return best_action
```

## ExpectimaxAgent
The `ExpectimaxAgent` class implements the expectimax algorithm, which models ghosts as choosing actions uniformly at random.

- **getAction**: Returns the expectimax action from the current game state.
- **expectimax**: Recursively calculates the best score for an agent using the expectimax algorithm.

```python
class ExpectimaxAgent(MultiAgentSearchAgent):
    def getAction(self, gameState):
        def expectimax(curr_depth, agent_index, gameState):
            if agent_index >= gameState.getNumAgents():
                agent_index = 0
                curr_depth += 1
            if curr_depth == self.depth:
                return self.evaluationFunction(gameState)
            if agent_index == 0:
                value = float('-inf')
                for action in gameState.getLegalActions(agent_index):
                    value = max(value, expectimax(curr_depth, agent_index + 1, gameState.generateSuccessor(agent_index, action)))
                return value
            else:
                value = 0
                legalActions = gameState.getLegalActions(agent_index)
                prob = 1 / len(legalActions)
                for action in legalActions:
                    value += prob * expectimax(curr_depth, agent_index + 1, gameState.generateSuccessor(agent_index, action))
                return value

        best_action = None
        value = float('-inf')
        for action in gameState.getLegalActions(0):
            new_value = expectimax(0, 1, gameState.generateSuccessor(0, action))
            if new_value > value:
                value = new_value
                best_action = action
        return best_action
```

## Evaluation Functions

### scoreEvaluationFunction
Returns the score of the current game state.

```python
def scoreEvaluationFunction(currentGameState):
    return currentGameState.getScore()
```

### betterEvaluationFunction
A more advanced evaluation function that considers the current score, distance to the closest food, distance to the closest ghost, and the number of remaining food pellets.

```python
def betterEvaluationFunction(currentGameState):
    pos = currentGameState.getPacmanPosition()
    food = currentGameState.getFood()
    ghostStates = currentGameState.getGhostStates()
    scaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]

    foodDistances = [manhattanDistance(pos, foodPos) for foodPos in food.asList()]
    closestFoodDistance = min(foodDistances) if foodDistances else 0

    ghostDistances = [manhattanDistance(pos, ghostState.getPosition()) for ghostState in ghostStates]
    closestGhostDistance = min(ghostDistances) if ghostDistances else 0

    score = currentGameState.getScore()
    score -= closestFoodDistance
    if closestGhostDistance > 0:
        score += max(closestGhostDistance, 5)
    return score

better = betterEvaluationFunction
```

## Conclusion

The multiAgents.py file provides a comprehensive implementation of various AI agents for the Pacman game, each using different strategies to optimize Pacman's performance. The reflex agent uses a simple evaluation function, while the minimax, alpha-beta, and expectimax agents use more advanced search algorithms to make decisions. The evaluation functions play a crucial role in determining the effectiveness of these agents.
