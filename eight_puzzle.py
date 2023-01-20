import numpy as np
from heapq import heappush, heappop
from animation import draw
import argparse
import math

class Node():
    """
    cost_from_start - the cost of reaching this node from the starting node
    state - the state (row,col)
    parent - the parent node of this node, default as None
    """
    def __init__(self, state, cost_from_start, parent = None):
        self.state = state
        self.parent = parent
        self.cost_from_start = cost_from_start


class EightPuzzle():
    
    def __init__(self, start_state, goal_state, method, algorithm, array_index):
        self.start_state = start_state
        self.goal_state = goal_state
        self.visited = [] # state
        self.method = method
        self.algorithm = algorithm
        self.m, self.n = start_state.shape 
        self.array_index = array_index
        

    def goal_test(self, current_state):
        if np.array_equal(current_state, [[1, 2, 3], [4, 5, 6], [7, 8, 0]]):
            return True
        return False

    def get_cost(self, current_state, next_state):
        return 1

    def get_successors(self, state):
        successors = []
        direction_rows = [0, 0, -1, 1]
        direction_columns = [-1, 1, 0, 0]
        position = np.where(state == 0)
        y, x = position[0][0], position[1][0]

        for k in range(4):
            if (x + direction_columns[k] <= 2 and y + direction_rows[k] >= 0) and (x + direction_columns[k] >= 0 and y +
                                                                                   direction_rows[k] <= 2):
                copy = state.copy()
                temp = copy[int(y) + direction_rows[k]][int(x) + direction_columns[k]]
                copy[int(y) + direction_rows[k]][int(x) + direction_columns[k]] = 0
                copy[int(y)][int(x)] = temp
                successors.append(copy)

        return successors

    # heuristics function
    def heuristics(self, state):
        cost = 0

        if self.method == 'Hamming':
            for row_num in range(len(state)):
                for column_num in range(len(state[row_num])):
                    if state[row_num][column_num] != self.goal_state[row_num][column_num]:
                        cost += 1

            return cost

        if self.method == 'Manhattan':
            correct_coordinates = {}
            for row_num in range(len(self.goal_state)):
                for column_num in range(len(self.goal_state[row_num])):
                    correct_coordinates[self.goal_state[row_num][column_num]] = [row_num, column_num]

            for row_num in range(len(state)):
                for column_num in range(len(state[row_num])):
                    correct_x_y = correct_coordinates[state[row_num][column_num]]
                    cost += abs(row_num - correct_x_y[0]) + abs(column_num - correct_x_y[1])

            return cost

    # priority of node 
    def priority(self, node):
        if self.algorithm == 'Greedy':
            return self.heuristics(node.state)
        elif self.algorithm == 'AStar':
            return self.heuristics(node.state) + node.cost_from_start

    # draw 
    def draw(self, node):
        path=[]
        while node.parent:
            path.append(node.state)
            node = node.parent
        path.append(self.start_state)

        draw(path[::-1], self.array_index, self.algorithm, self.method)

    # solve it
    def solve(self):
        # use one framework to merge all four algorithms.
        # !!! In A* algorithm, you only need to return the first solution. 
        #     The first solution is in general possibly not the best solution, however, in this eight puzzle, 
        #     we can prove that the first solution is the best solution. 
        # your code goes here:
        if self.goal_test(self.start_state):
            return

        state = self.start_state.copy()
        self.visited.append(state)
        first_node = Node(state, 0, None)
        index = 0
        priority_queue = [(self.priority(first_node), index, first_node)]

        while priority_queue:
            best_node = heappop(priority_queue)[2]

            successors = self.get_successors(best_node.state)

            for successor in successors:
                visited = False
                for visited_state in self.visited:
                    if np.array_equal(visited_state, successor):
                        visited = True

                if visited:
                    continue

                self.visited.append(successor)

                next_node = Node(successor, best_node.cost_from_start + 1, best_node)
                if self.goal_test(successor):
                    self.draw(next_node)
                    return
                index += 1
                heappush(priority_queue, (self.priority(next_node), index, next_node))


if __name__ == "__main__":
    
    goal = np.array([[1,2,3],[4,5,6],[7,8,0]])
    start_arrays = [np.array([[1,2,0],[3,4,6],[7,5,8]]),
                    np.array([[8,1,3],[4,0,2],[7,6,5]])]
    methods = ["Hamming", "Manhattan"]
    algorithms = ['Depth-Limited-DFS', 'BFS', 'Greedy', 'AStar']
    
    parser = argparse.ArgumentParser(description='eight puzzle')

    parser.add_argument('-array', dest='array_index', required = True, type = int, help='index of array')
    parser.add_argument('-method', dest='method_index', required = True, type = int, help='index of method')
    parser.add_argument('-algorithm', dest='algorithm_index', required = True, type = int, help='index of algorithm')

    args = parser.parse_args()

    # Example:
    # Run this in the terminal using array 0, method Hamming, algorithm AStar:
    #     python eight_puzzle.py -array 0 -method 0 -algorithm 3
    game = EightPuzzle(start_arrays[args.array_index], goal, methods[args.method_index], algorithms[args.algorithm_index], args.array_index)
    game.solve()