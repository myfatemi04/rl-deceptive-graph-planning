from abc import ABC, abstractmethod
from functools import cached_property, lru_cache

import networkx as nx
import numpy as np


def get_goal_probabilities(goal_value_function_dict, goal_probability_dict, start_pos, curr_pos):
    probability_weight_dict = {
        goal: goal_probability_dict.get(goal, 0) * np.exp(
            goal_value_function_dict[goal][curr_pos] - goal_value_function_dict[goal][start_pos]
        )
        for goal in goal_value_function_dict.keys()
    }

    denominator = sum(probability_weight_dict.values())

    probabilities = {
        goal: probability_weight_dict[goal] / denominator
        for goal in goal_value_function_dict.keys()
    }

    return probabilities

class PathBonusFunction(ABC):
    @abstractmethod
    def __call__(self, path) -> float:
        pass

    def __add__(self, other):
        assert isinstance(other, PathBonusFunction), "Can only add PathBonusFunctions to other PathBonusFunctions."
        return BonusFunctionCombination([self, other], [1, 1])
    
    def __mul__(self, scalar):
        assert isinstance(scalar, (int, float)), "Can only multiply PathBonusFunctions by scalars."
        return BonusFunctionCombination([self], [scalar])

class BonusFunctionCombination(PathBonusFunction):
    def __init__(self, bonus_function_list, bonus_function_weights):
        self.bonus_function_list = bonus_function_list
        self.bonus_function_weights = bonus_function_weights

    def __call__(self, path):
        return sum(bonus_function(path) * weight for bonus_function, weight in zip(self.bonus_function_list, self.bonus_function_weights))

class AmbiguousDeceptionFunctionV1(PathBonusFunction):
    def __init__(self, goal_value_function_dict, goal_probability_dict, target_goal, gamma):
        self.goal_value_function_dict = goal_value_function_dict
        self.goal_probability_dict = goal_probability_dict
        self.target_goal = target_goal
        self.gamma = gamma

        assert target_goal in goal_probability_dict.keys(), "Target goal must be in goal probability dict."
        assert target_goal in goal_value_function_dict.keys(), "Target goal must be in goal value function dict."

    def _get_goal_dissimilarity(self, start_pos, curr_pos):
        other_goals = set(self.goal_value_function_dict.keys())
        other_goals.remove(self.target_goal)

        goal_probs = get_goal_probabilities(self.goal_value_function_dict, self.goal_probability_dict, start_pos, curr_pos)

        # print(goal_probs)

        # Eq. (6) of "Deceptive Decision-Making Under Uncertainty"
        # entropy = sum(prob * (0.1 + np.log(prob + 1e-5)) for prob in goal_probs.values())
        dissimilarity = 0

        for goal_a in self.goal_probability_dict.keys():
            for goal_b in self.goal_probability_dict.keys():
                dissimilarity += self.goal_probability_dict[goal_a] * self.goal_probability_dict[goal_b] * abs(goal_probs[goal_a] - goal_probs[goal_b])

        return dissimilarity

    def __call__(self, path):
        if len(path) == 1:
            return 0
        
        base_dissimilarity = self._get_goal_dissimilarity(path[0], path[0])
        curr_dissimilarity = self._get_goal_dissimilarity(path[0], path[-1])
        prev_dissimilarity = self._get_goal_dissimilarity(path[0], path[-2])
        total_potential_change = self._get_goal_dissimilarity(path[0], self.target_goal) - base_dissimilarity

        # "How much less ambiguous are we now, compared to before?"
        # Scale by the total amount that this ambiguity score will ever change, so the potential
        # sums to 1 (or in this case, negative 1).
        # change = (curr_dissimilarity - prev_dissimilarity) * 5
        change = curr_dissimilarity
        print(f"{change:.3f} {self.gamma ** len(path):.3f}")

        return -change * self.gamma ** len(path)
    
class AmbiguousDeceptionFunctionV2(PathBonusFunction):
    def __init__(self, goal_value_function_dict, goal_probability_dict, target_goal, gamma):
        self.goal_value_function_dict = goal_value_function_dict
        self.goal_probability_dict = goal_probability_dict
        self.target_goal = target_goal
        self.gamma = gamma

        # max_possible_dissimilarity = -np.inf
        # max_possible_dissimilarity_pos = None
        # min_possible_dissimilarity = np.inf
        # min_possible_dissimilarity_pos = None

        # all_nodes = list(self.goal_value_function_dict[self.target_goal].keys())

        # for start_pos in all_nodes:
        #     for final_pos in all_nodes:
        #         if start_pos == final_pos:
        #             continue

        #         dissimilarity = self._get_goal_dissimilarity(start_pos, final_pos)
        #         if dissimilarity > max_possible_dissimilarity:
        #             max_possible_dissimilarity = dissimilarity
        #             max_possible_dissimilarity_pos = (start_pos, final_pos)
        #         if dissimilarity < min_possible_dissimilarity:
        #             min_possible_dissimilarity = dissimilarity
        #             min_possible_dissimilarity_pos = (start_pos, final_pos)
        
        # print(max_possible_dissimilarity, max_possible_dissimilarity_pos)
        # print(min_possible_dissimilarity, min_possible_dissimilarity_pos)

        assert target_goal in goal_probability_dict.keys(), "Target goal must be in goal probability dict."
        assert target_goal in goal_value_function_dict.keys(), "Target goal must be in goal value function dict."

    @lru_cache(maxsize=128)
    def max_possible_dissimilarity(self):
        max_possible_dissimilarity = -np.inf
        for goal_a in self.goal_probability_dict.keys():
            for goal_b in self.goal_probability_dict.keys():
                if goal_a == goal_b:
                    continue

                dissimilarity = self._get_goal_dissimilarity(goal_a, goal_b)
                if dissimilarity > max_possible_dissimilarity:
                    max_possible_dissimilarity = dissimilarity
        
        return max_possible_dissimilarity

    def _get_goal_dissimilarity(self, start_pos, curr_pos):
        other_goals = set(self.goal_value_function_dict.keys())
        other_goals.remove(self.target_goal)

        goal_probs = get_goal_probabilities(self.goal_value_function_dict, self.goal_probability_dict, start_pos, curr_pos)

        # Eq. (6) of "Deceptive Decision-Making Under Uncertainty"
        # entropy = sum(prob * (0.1 + np.log(prob + 1e-5)) for prob in goal_probs.values())
        dissimilarity = 0

        for goal_a in self.goal_probability_dict.keys():
            for goal_b in self.goal_probability_dict.keys():
                dissimilarity += self.goal_probability_dict[goal_a] * self.goal_probability_dict[goal_b] * abs(goal_probs[goal_a] - goal_probs[goal_b])

        # Divide by 2 for symmetry
        return dissimilarity / 2

    def __call__(self, path, graph):
        if len(path) == 1:
            return 0
        
        curr_dissimilarity = self._get_goal_dissimilarity(path[0], path[-1])
        # max_dissimilarity = 

        goal_distance = nx.shortest_path_length(graph, path[-1], self.target_goal)
        other_goals = set(self.goal_probability_dict.keys())
        other_goals.remove(self.target_goal)
        decoy = other_goals.pop()
        decoy_distance = nx.shortest_path_length(graph, path[-1], decoy)
        goal_decoy_distance = nx.shortest_path_length(graph, self.target_goal, decoy)

        # Scale by the total amount that this ambiguity score will ever change, so the potential
        # sums to 1 (or in this case, negative 1).
        # change = (curr_dissimilarity - prev_dissimilarity) * 5
        value = -curr_dissimilarity * (0.5 ** (((goal_distance + decoy_distance) / 2) / (goal_decoy_distance)))
        # print(f"{value:.3f} {self.gamma ** len(path):.3f} {max(goal_distance, decoy_distance)}")

        print(curr_dissimilarity, value, self.gamma ** len(path), ((goal_distance + decoy_distance) / 2) / (goal_decoy_distance))

        return value * self.gamma ** len(path)

class AmbiguousDeceptionFunctionV3(PathBonusFunction):
    def __init__(self, goal_value_function_dict, goal_probability_dict, target_goal, gamma):
        self.goal_value_function_dict = goal_value_function_dict
        self.goal_probability_dict = goal_probability_dict
        self.target_goal = target_goal
        self.gamma = gamma

        assert target_goal in goal_probability_dict.keys(), "Target goal must be in goal probability dict."
        assert target_goal in goal_value_function_dict.keys(), "Target goal must be in goal value function dict."

    @cached_property
    def max_possible_unambiguity(self):
        # when using absolute values, the highest possible dissimilarity is to have one goal be 1 and the others be 0
        # the value of this will be sum(other goal probabilities) * (my goal probability)
        # because the sums will expand to P_a * P_a * (1 - 1) + P_a * P_b * (1 - 0) + P_a * P_c * (1 - 0) + ...
        # which is just P_a * (P_b + P_c + ...)
        max_possible_dissimilarity = -np.inf
        for goal in self.goal_probability_dict.keys():
            prob = self.goal_probability_dict[goal]
            possible_dissimilarity = prob * (1 - prob)
            if possible_dissimilarity > max_possible_dissimilarity:
                max_possible_dissimilarity = possible_dissimilarity
        
        return max_possible_dissimilarity

    def _get_goal_dissimilarity(self, start_pos, curr_pos):
        other_goals = set(self.goal_value_function_dict.keys())
        other_goals.remove(self.target_goal)

        goal_probs = get_goal_probabilities(self.goal_value_function_dict, self.goal_probability_dict, start_pos, curr_pos)

        # Eq. (6) of "Deceptive Decision-Making Under Uncertainty"
        # entropy = sum(prob * (0.1 + np.log(prob + 1e-5)) for prob in goal_probs.values())
        dissimilarity = 0

        for goal_a in self.goal_probability_dict.keys():
            for goal_b in self.goal_probability_dict.keys():
                if goal_a == goal_b:
                    continue
                dissimilarity += self.goal_probability_dict[goal_a] * self.goal_probability_dict[goal_b] * abs(goal_probs[goal_a] - goal_probs[goal_b])

        # Divide by 2 for symmetry
        return dissimilarity / 2

    def __call__(self, path, graph):
        if len(path) == 1:
            return 0
        
        unambiguity_unscaled = self._get_goal_dissimilarity(path[0], path[-1])
        unambiguity = unambiguity_unscaled / self.max_possible_unambiguity
        ambiguity = 1 - unambiguity

        # Decay based on distance to other goals
        # Similar to adding a potential that shapes towards the goal
        goal_distance = nx.shortest_path_length(graph, path[-1], self.target_goal)
        initial_goal_distance = nx.shortest_path_length(graph, path[0], self.target_goal)
        other_goals = set(self.goal_probability_dict.keys())
        other_goals.remove(self.target_goal)
        decoy = other_goals.pop()
        decoy_distance = nx.shortest_path_length(graph, path[-1], decoy)
        distance_between_true_and_decoy = nx.shortest_path_length(graph, self.target_goal, decoy)

        # Deceptiveness scales down the longer the path becomes.
        # For static experiments, just take the shortest possible path length.
        LP = nx.shortest_path_length(graph, path[0], path[-1])
        deceptiveness_mix = self.gamma ** LP
        average_distance_to_both = (goal_distance + decoy_distance) / 2
        relative_distance_from_both_goals = (average_distance_to_both / (distance_between_true_and_decoy / 2) - 1)
        relative_distance_from_goal = goal_distance / (distance_between_true_and_decoy / 2)

        # Bonus for being close to both goals: 0.5 ^ (average distance to either goal) / (minimum average distance to both goals)
        goal_proximity_bonus = 0.2 ** relative_distance_from_goal
        either_goal_proximity_bonus = 0.2 ** relative_distance_from_both_goals
        # value = (ambiguity) * deceptiveness_mix + goal_proximity_bonus * (1 - deceptiveness_mix)

        distance_scale = (distance_between_true_and_decoy - abs(goal_distance - decoy_distance)) / (distance_between_true_and_decoy)
        value = distance_scale * deceptiveness_mix + goal_proximity_bonus * (1 - deceptiveness_mix)

        return value

        # return value

class AmbiguousDeceptionFunctionV4(PathBonusFunction):
    def __init__(self, goal_value_function_dict, goal_probability_dict, target_goal, gamma, graph):
        self.goal_value_function_dict = goal_value_function_dict
        self.goal_probability_dict = goal_probability_dict
        self.target_goal = target_goal
        self.gamma = gamma
        self.graph = graph

        assert target_goal in goal_probability_dict.keys(), "Target goal must be in goal probability dict."
        assert target_goal in goal_value_function_dict.keys(), "Target goal must be in goal value function dict."

    def __call__(self, path):
        if len(path) == 1:
            return 0

        # Decay based on distance to other goals
        # Similar to adding a potential that shapes towards the goal
        goal_distance = nx.shortest_path_length(self.graph, path[-1], self.target_goal)
        other_goals = set(self.goal_probability_dict.keys())
        other_goals.remove(self.target_goal)
        decoy = other_goals.pop()
        decoy_distance = nx.shortest_path_length(self.graph, path[-1], decoy)
        distance_between_true_and_decoy = nx.shortest_path_length(self.graph, self.target_goal, decoy)

        # print(decoy)

        distance_scale = (distance_between_true_and_decoy - abs(goal_distance - decoy_distance)) / distance_between_true_and_decoy
        # return abs(goal_distance - decoy_distance)
        # return 1 if decoy_distance == 4 else 0
        return distance_scale * self.gamma ** len(path)

class AmbiguousDeceptionFunctionV5(PathBonusFunction):
    def __init__(self, goal_value_function_dict, goal_probability_dict, target_goal, gamma, graph):
        self.goal_value_function_dict = goal_value_function_dict
        self.goal_probability_dict = goal_probability_dict
        self.target_goal = target_goal
        self.gamma = gamma
        self.graph = graph

        assert target_goal in goal_probability_dict.keys(), "Target goal must be in goal probability dict."
        assert target_goal in goal_value_function_dict.keys(), "Target goal must be in goal value function dict."

    def __call__(self, path):
        if len(path) == 1:
            return 0
        
        # Don't allow cycles
        if path[-1] in path[:-1]:
            return 0

        # Decay based on distance to other goals
        # Similar to adding a potential that shapes towards the goal
        goal_distance = nx.shortest_path_length(self.graph, path[-1], self.target_goal)
        other_goals = set(self.goal_probability_dict.keys())
        other_goals.remove(self.target_goal)
        decoy = other_goals.pop()
        decoy_distance = nx.shortest_path_length(self.graph, path[-1], decoy)
        distance_between_true_and_decoy = nx.shortest_path_length(self.graph, self.target_goal, decoy)

        distance_scale = (distance_between_true_and_decoy - abs(goal_distance - decoy_distance)) / distance_between_true_and_decoy
        
        return distance_scale * self.gamma ** len(path)

AmbiguousDeceptionFunction = AmbiguousDeceptionFunctionV5

class MinimizeLikelihoodDeceptionFunction(PathBonusFunction):
    def __init__(self, goal_value_function_dict, goal_probability_dict, target_goal, gamma):
        self.goal_value_function_dict = goal_value_function_dict
        self.goal_probability_dict = goal_probability_dict
        self.target_goal = target_goal
        self.gamma = gamma

        assert target_goal in goal_probability_dict.keys(), "Target goal must be in goal probability dict."
        assert target_goal in goal_value_function_dict.keys(), "Target goal must be in goal value function dict."

    def __call__(self, path):
        other_goals = set(self.goal_value_function_dict.keys())
        other_goals.remove(self.target_goal)

        goal_probs = get_goal_probabilities(self.goal_value_function_dict, self.goal_probability_dict, path[0], path[-1])

        return (1 - goal_probs[self.target_goal]) * self.gamma ** len(path)
    

class Differentiate(PathBonusFunction):
    def __init__(self, function):
        self.function = function
    
    def __call__(self, path):
        if len(path) == 0:
            return 0
        if len(path) == 1:
            return self.function(path)
        else:
            return self.function(path) - self.function(path[:-1])

# Adds a bonus depending on distance to goal.
class GoalBonusFunction(PathBonusFunction):
    def __init__(self, graph: nx.Graph, target_goal, gamma: float):
        self.graph = graph
        self.target_goal = target_goal
        self.gamma = gamma

    def __call__(self, path):
        # Calculate the "advantage" by comparing to all alternative actions.
        # Normalize by dividing by the standard deviation. If all actions are
        # equivalent, then the advantage will be zero before this division.
        # Nevertheless, we add a small "epsilon" term to prevent NaN's from
        # appearing.
        if len(path) < 2:
            return 0
        
        prev_distance = nx.shortest_path_length(self.graph, path[-2], self.target_goal)
        this_distance = nx.shortest_path_length(self.graph, path[-1], self.target_goal)

        advantage = prev_distance - this_distance

        return advantage

class ExaggerationDeceptionFunctionV1(PathBonusFunction):
    def __init__(self, goal_value_function_dict: dict, goal_probability_dict: dict, target_goal, gamma: float):
        self.goal_value_function_dict = goal_value_function_dict
        self.goal_probability_dict = goal_probability_dict
        self.target_goal = target_goal
        self.gamma = gamma

        assert target_goal in goal_probability_dict.keys(), "Target goal must be in goal probability dict."
        assert target_goal in goal_value_function_dict.keys(), "Target goal must be in goal value function dict."

    def __call__(self, path):
        # Don't allow cycles
        if path[-1] in path[:-1]:
            return 0

        other_goals = set(self.goal_probability_dict.keys())
        other_goals.remove(self.target_goal)
        
        goal_probs = get_goal_probabilities(self.goal_value_function_dict, self.goal_probability_dict, path[0], path[-1])
        # print([goal_probs[g] for g in self.goal_probability_dict])
        target_goal_prob = goal_probs.pop(self.target_goal)

        # Eq. (5) of "Deceptive Decision-Making Under Uncertainty"
        deceptiveness = -1 * (target_goal_prob - max(goal_probs.values()))

        return deceptiveness * self.gamma ** len(path)

class ExaggerationDeceptionFunctionV2(PathBonusFunction):
    def __init__(self, goal_value_function_dict: dict, goal_probability_dict: dict, target_goal, gamma: float, graph: nx.Graph):
        self.goal_value_function_dict = goal_value_function_dict
        self.goal_probability_dict = goal_probability_dict
        self.target_goal = target_goal
        self.gamma = gamma
        self.graph = graph

        assert target_goal in goal_probability_dict.keys(), "Target goal must be in goal probability dict."
        assert target_goal in goal_value_function_dict.keys(), "Target goal must be in goal value function dict."

    def __call__(self, path):
        # Don't allow cycles
        if path[-1] in path[:-1]:
            return 0

        other_goals = set(self.goal_probability_dict.keys())
        other_goals.remove(self.target_goal)
        decoy_goal = other_goals.pop()
        distance_to_decoy = nx.shortest_path_length(self.graph, path[-1], decoy_goal)
        distance_to_target = nx.shortest_path_length(self.graph, path[-1], self.target_goal)
        distance_between_goals = nx.shortest_path_length(self.graph, decoy_goal, self.target_goal)

        deceptiveness = -(distance_to_decoy - distance_to_target) / distance_between_goals

        return deceptiveness * self.gamma ** len(path)

ExaggerationDeceptionFunction = ExaggerationDeceptionFunctionV1

class StaticMovementPenalty(PathBonusFunction):
    def __init__(self, penalty_amount):
        pass

    def __call__(self, path):
        return 

# Introduced to help models move towards the goal in large environments,
# with version 26 (geometric distance testing, without deceptiveness yet).
class RewardMovingToGoalNondeceptiveFunction(PathBonusFunction):
    def __init__(self, distance_matrix, target_goal, gamma: float = 1):
        self.distance_matrix = distance_matrix
        self.target_goal = target_goal
        self.gamma = gamma

    def __call__(self, path):
        # Calculate the "advantage" by comparing to all alternative actions.
        # Normalize by dividing by the standard deviation. If all actions are
        # equivalent, then the advantage will be zero before this division.
        # Nevertheless, we add a small "epsilon" term to prevent NaN's from
        # appearing.
        if len(path) < 2:
            return 0
        
        prev_distance = self.distance_matrix[self.target_goal][path[-2]]
        this_distance = self.distance_matrix[self.target_goal][path[-1]]

        advantage = prev_distance - this_distance

        return advantage * self.gamma ** len(path)
