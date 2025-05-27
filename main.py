import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Dict
import random
from dataclasses import dataclass
import time
import argparse # Added
import os # Added

# Ensure matplotlib backend is non-interactive if plots are not shown
# This should be set before importing pyplot if there's a chance of no display.
# However, for conditional showing, it's better to manage plt.show() calls.

@dataclass
class ACOResult:
    """ACO算法结果数据类"""

    best_path: Optional[List[int]]
    best_distance: float
    convergence_history: List[float]
    iteration_count: int
    execution_time: float
    problem_name: str = "TSP" # Added for better reporting
    algorithm_name: str = "ACO" # Added for better reporting


class BaseTSP(ABC):
    """TSP问题基类"""

    def __init__(
        self, distance_matrix: np.ndarray, city_names: Optional[List[str]] = None
    ):
        self.distance_matrix = np.array(distance_matrix)
        self.n_cities = len(distance_matrix)
        self.city_names = city_names or [f"City_{i}" for i in range(self.n_cities)]
        self._validate_distance_matrix()

    def _validate_distance_matrix(self):
        """验证距离矩阵的有效性"""
        if self.distance_matrix.shape != (self.n_cities, self.n_cities):
            raise ValueError("距离矩阵必须是方阵")
        if np.any(self.distance_matrix < 0):
            raise ValueError("距离不能为负数")
        # if np.any(np.diag(self.distance_matrix) != 0): # This might be too strict if the matrix generation allows it
        #     # Check if all diagonal elements are zero
        #     for i in range(self.n_cities):
        #         if self.distance_matrix[i,i] != 0:
        #             print(f"Warning: Distance from city {i} to itself is {self.distance_matrix[i,i]}, expected 0.")
        #             # raise ValueError("城市到自身的距离应为0") # Relaxing this for now
        pass


    @abstractmethod
    def is_symmetric(self) -> bool:
        """判断是否为对称TSP"""
        pass

    def calculate_tour_distance(self, tour: List[int]) -> float:
        """计算旅行路径的总距离"""
        if not tour: # Handle empty tour case
            return float('inf')
        # if len(set(tour)) != self.n_cities: # Check if all cities are visited exactly once
            # This check is more for a valid TSP tour; ACO construction should ensure this.
            # For partial calculations or debugging, this might be too strict.
            # Let's assume tour is a permutation for now.
            # pass

        total_distance = 0.0
        for i in range(len(tour)):
            current_city = tour[i]
            next_city = tour[(i + 1) % len(tour)] # Handles wrap-around for the last city to the first
            total_distance += self.distance_matrix[current_city][next_city]

        return total_distance

    def get_distance(self, from_city: int, to_city: int) -> float:
        """获取两城市间距离"""
        return self.distance_matrix[from_city][to_city]

    def generate_random_cities( # This method seems unused in the provided classes, but kept for completeness
        self, n_cities: int, coordinate_range: Tuple[float, float] = (0, 100)
    ) -> np.ndarray:
        """生成随机城市坐标"""
        # np.random.seed(42) # Seed should be controlled globally if needed for reproducibility
        return np.random.uniform(
            coordinate_range[0], coordinate_range[1], (n_cities, 2)
        )


class SymmetricTSP(BaseTSP):
    """对称TSP问题类"""

    def __init__(
        self,
        distance_matrix: Optional[np.ndarray] = None,
        city_coordinates: Optional[np.ndarray] = None,
        city_names: Optional[List[str]] = None,
    ):
        if distance_matrix is not None:
            super().__init__(distance_matrix, city_names)
            if city_coordinates is not None: # Store coordinates if provided with matrix
                 self.coordinates = city_coordinates
            else: # Try to infer coordinates if needed for visualization, or mark as unavailable
                 self.coordinates = None
        elif city_coordinates is not None:
            distance_matrix = self._calculate_euclidean_distances(city_coordinates)
            super().__init__(distance_matrix, city_names)
            self.coordinates = city_coordinates
        else:
            raise ValueError("必须提供距离矩阵或城市坐标")
        
        # Ensure diagonal is zero for symmetric TSP from coordinates
        if hasattr(self, 'coordinates') and self.coordinates is not None: # check coordinates exist
             if self.distance_matrix is not None : # check distance_matrix exist
                np.fill_diagonal(self.distance_matrix, 0)


    def _calculate_euclidean_distances(self, coordinates: np.ndarray) -> np.ndarray:
        """根据坐标计算欧几里得距离矩阵"""
        n = len(coordinates)
        distances = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    distances[i][j] = np.sqrt(
                        np.sum((coordinates[i] - coordinates[j]) ** 2)
                    )
        return distances

    def is_symmetric(self) -> bool:
        return True

    @classmethod
    def create_random_instance(
        cls, n_cities: int, coordinate_range: Tuple[Tuple[float, float], Tuple[float, float]] = ((0, 100), (0, 100)), seed: Optional[int] = None
    ):
        """创建随机对称TSP实例"""
        if seed is not None:
            np.random.seed(seed)
        coordinates = np.zeros((n_cities, 2))
        coordinates[:, 0] = np.random.uniform(coordinate_range[0][0], coordinate_range[0][1], n_cities)
        coordinates[:, 1] = np.random.uniform(coordinate_range[1][0], coordinate_range[1][1], n_cities)
        # Pass coordinates to constructor so they are stored
        return cls(city_coordinates=coordinates)
    # @classmethod
    # def create_random_instance(
    #     cls, n_cities: int, coordinate_range: Tuple[float, float] = (0, 100), seed: Optional[int] = None
    # ):
    #     """创建随机对称TSP实例"""
    #     if seed is not None:
    #         np.random.seed(seed)
    #     coordinates = np.random.uniform(
    #         coordinate_range[0], coordinate_range[1], (n_cities, 2)
    #     )
    #     # Pass coordinates to constructor so they are stored
    #     return cls(city_coordinates=coordinates)


class AsymmetricTSP(BaseTSP):
    """非对称TSP问题类"""

    def __init__(
        self, distance_matrix: np.ndarray, city_names: Optional[List[str]] = None
    ):
        super().__init__(distance_matrix, city_names)
        # Ensure diagonal is zero
        if self.distance_matrix is not None : # check distance_matrix exist
            np.fill_diagonal(self.distance_matrix, 0)


    def is_symmetric(self) -> bool:
        return False

    @classmethod
    def create_random_instance(cls, n_cities: int, asymmetry_factor: float = 0.3, seed: Optional[int] = None):
        """创建随机非对称TSP实例"""
        if seed is not None:
            np.random.seed(seed)
        # 先生成对称矩阵
        coordinates = np.random.uniform(0, 100, (n_cities, 2)) # Base for distances
        base_distances = np.zeros((n_cities, n_cities))
        for i in range(n_cities):
            for j in range(n_cities):
                if i != j:
                    base_distances[i][j] = np.sqrt(
                        np.sum((coordinates[i] - coordinates[j]) ** 2)
                    )

        # 添加非对称性
        asymmetric_distances = base_distances.copy()
        for i in range(n_cities):
            for j in range(n_cities):
                if i != j:
                    # 随机调整距离以创造非对称性
                    factor = 1 + np.random.uniform(-asymmetry_factor, asymmetry_factor)
                    asymmetric_distances[i][j] *= factor
        
        np.fill_diagonal(asymmetric_distances, 0) # Ensure diagonal is zero
        return cls(asymmetric_distances)


class BaseACO(ABC):
    """蚁群算法基类"""

    def __init__(
        self,
        tsp_problem: BaseTSP,
        n_ants: int = 20,
        alpha: float = 1.0,
        beta: float = 2.0,
        evaporation_rate: float = 0.1,
        initial_pheromone: float = 0.1,
        max_iterations: int = 100,
        q0: float = 0.0, # Parameter for ACS (exploration/exploitation)
        seed: Optional[int] = None
    ):
        self.tsp = tsp_problem
        self.n_ants = n_ants
        self.alpha = alpha
        self.beta = beta
        self.evaporation_rate = evaporation_rate
        self.initial_pheromone = initial_pheromone
        self.max_iterations = max_iterations
        self.q0 = q0 # For ACS or similar variants that use explicit exploitation probability

        if seed is not None:
            random.seed(seed)
            # np.random.seed(seed) # If numpy random is used within ACO itself

        self.pheromone_matrix = np.full(
            (self.tsp.n_cities, self.tsp.n_cities), initial_pheromone, dtype=float
        )
        np.fill_diagonal(self.pheromone_matrix, 0) # No pheromone on path from i to i

        self.heuristic_matrix = np.zeros((self.tsp.n_cities, self.tsp.n_cities))
        for i in range(self.tsp.n_cities):
            for j in range(self.tsp.n_cities):
                if i != j:
                    dist = self.tsp.get_distance(i, j)
                    if dist == 0: # Avoid division by zero if cities are coincident
                        # This implies infinite heuristic. Handle carefully in selection.
                        # A very large number can simulate infinity if direct inf causes issues.
                        self.heuristic_matrix[i][j] = 1e18 # A large number
                    else:
                        self.heuristic_matrix[i][j] = 1.0 / dist
        
        self.best_global_tour: Optional[List[int]] = None
        self.best_global_distance = float("inf")
        self.convergence_history: List[float] = []
        self.iteration_count_for_solve = 0


    def _construct_ant_solution(self, ant_id: int) -> Tuple[List[int], float]:
        """构造单只蚂蚁的解"""
        start_city = random.randint(0, self.tsp.n_cities - 1)
        tour = [start_city]
        # Use a list for unvisited to preserve order for random.choice if needed,
        # but convert to set for faster removal checking if performance is critical for large N.
        # For typical N in TSP for ACO, list operations are usually fine.
        unvisited = list(range(self.tsp.n_cities))
        unvisited.remove(start_city)
        # random.shuffle(unvisited) # Optional: shuffle to break ties differently if selection yields multiple same-prob options

        current_city = start_city
        while unvisited:
            next_city = self._select_next_city(current_city, unvisited)
            if next_city is None: # Should ideally not happen if unvisited is not empty
                 # Fallback: pick a random unvisited city. This indicates an issue in _select_next_city or logic.
                 # print(f"Warning: _select_next_city returned None for ant {ant_id} from city {current_city} with unvisited: {unvisited}. Choosing randomly.")
                 if unvisited: next_city = random.choice(unvisited)
                 else: break # Should not be reached if loop condition is `while unvisited`
            tour.append(next_city)
            unvisited.remove(next_city) # Efficient for list if unvisited is small, for large N, a set might be better for remove.
            current_city = next_city
        
        if len(tour) == self.tsp.n_cities: # Ensure a full tour
            distance = self.tsp.calculate_tour_distance(tour)
            return tour, distance
        else: # Should not happen in normal operation if logic is correct
            # print(f"Warning: Ant {ant_id} constructed an incomplete tour of length {len(tour)} for {self.tsp.n_cities} cities.")
            return tour, float('inf') # Penalize incomplete tours heavily


    def _select_next_city(self, current_city: int, unvisited: List[int]) -> Optional[int]:
        """选择下一个城市"""
        if not unvisited:
            return None

        # ACS-style exploration/exploitation choice based on q0
        if self.q0 > 0 and random.random() < self.q0:
            # Exploitation: choose the city with max (pheromone^alpha * heuristic^beta)
            max_product_value = -1.0
            selected_city_exploit = -1 # Initialize to an invalid city index
            
            # Find city with the highest product value
            for city_candidate in unvisited:
                pheromone_val = self.pheromone_matrix[current_city][city_candidate] ** self.alpha
                heuristic_val = self.heuristic_matrix[current_city][city_candidate] ** self.beta
                current_product_value = pheromone_val * heuristic_val
                
                if current_product_value > max_product_value:
                    max_product_value = current_product_value
                    selected_city_exploit = city_candidate
                # Tie-breaking: if product values are equal, can add random choice among ties
                # For simplicity, current code takes the first one found or last one that strictly improves.

            if selected_city_exploit != -1 : # If a city was found
                 return selected_city_exploit
            else: # Fallback if all values are zero/negative (unlikely) or no unvisited (caught by first line)
                return random.choice(unvisited) if unvisited else None


        # Exploration: roulette wheel selection
        probabilities = []
        # Create a list of (city, prob_value) to handle potential issues with inf/nan more easily before normalization
        city_prob_values = []

        for city_candidate in unvisited:
            pheromone_val = self.pheromone_matrix[current_city][city_candidate] ** self.alpha
            heuristic_val = self.heuristic_matrix[current_city][city_candidate] ** self.beta
            prob_value = pheromone_val * heuristic_val
            city_prob_values.append((city_candidate, prob_value))

        # Filter out non-finite values or handle them. For now, sum finite values.
        # If heuristic was 1e18 for zero distance, product can be very large.
        finite_prob_values = [pv for city, pv in city_prob_values if np.isfinite(pv) and pv > 0]
        finite_cities = [city for city, pv in city_prob_values if np.isfinite(pv) and pv > 0]

        if not finite_cities: # No valid finite positive probabilities
            # This can happen if all pheromones are near zero and heuristics are also small or zero.
            # Or if all lead to non-finite products (e.g. inf*0 = nan)
            # Fallback to random choice among all unvisited cities.
            return random.choice(unvisited) if unvisited else None
            
        total_prob = sum(finite_prob_values)

        if total_prob == 0: # All finite_prob_values were zero (or negative, though unlikely for P*H)
            return random.choice(finite_cities) if finite_cities else (random.choice(unvisited) if unvisited else None)

        # Normalize probabilities
        normalized_probabilities = [p / total_prob for p in finite_prob_values]
        
        # Roulette wheel selection
        r = random.random()
        cumulative_prob = 0.0
        for i, prob_norm in enumerate(normalized_probabilities):
            cumulative_prob += prob_norm
            if r <= cumulative_prob:
                return finite_cities[i]
        
        # Fallback: if loop finishes (e.g., floating point issues, or r is extremely close to 1.0)
        # Return the last city in the list for which a probability was calculated.
        return finite_cities[-1] if finite_cities else (random.choice(unvisited) if unvisited else None)


    def _evaporate_pheromones(self):
        """信息素蒸发"""
        self.pheromone_matrix *= (1.0 - self.evaporation_rate)
        # Optional: set a minimum pheromone level (tau_min) to prevent premature stagnation
        # tau_min = 0.01 # Example value, should be configurable
        # self.pheromone_matrix[self.pheromone_matrix < tau_min] = tau_min
        # np.fill_diagonal(self.pheromone_matrix, 0) # Ensure diagonal remains 0


    @abstractmethod
    def _deposit_pheromones(self, ant_tours: List[Tuple[List[int], float]]):
        """信息素沉积 - 由子类实现"""
        pass

    def solve(self, problem_name="TSP", algorithm_name="ACO_Base") -> ACOResult:
        """求解TSP问题"""
        start_time = time.time()
        self.best_global_tour = None 
        self.best_global_distance = float("inf")
        self.convergence_history = [] # Stores best_global_distance at each iteration

        for iteration in range(self.max_iterations):
            ant_tours: List[Tuple[List[int], float]] = []
            # iteration_best_distance = float('inf') # For MMAS or iteration-best updates
            # iteration_best_tour = None

            for ant_idx in range(self.n_ants):
                tour, distance = self._construct_ant_solution(ant_idx)
                if tour and len(tour) == self.tsp.n_cities : # Ensure valid tour
                    ant_tours.append((tour, distance))
                    if distance < self.best_global_distance:
                        self.best_global_distance = distance
                        self.best_global_tour = tour[:] 
                    # if distance < iteration_best_distance: 
                    #     iteration_best_distance = distance
                    #     iteration_best_tour = tour[:]
            
            if not ant_tours: 
                # print(f"Warning: No valid tours constructed in iteration {iteration + 1}")
                # Append previous best if no new tours, or a very high value if first iteration and no tours
                current_best_to_log = self.best_global_distance if self.best_global_distance != float('inf') else (self.convergence_history[-1] if self.convergence_history else 0)
                self.convergence_history.append(current_best_to_log)
                self.iteration_count_for_solve = iteration + 1
                if self._check_early_stopping(iteration): break
                continue # Skip pheromone update if no ants completed tours

            self._evaporate_pheromones()
            self._deposit_pheromones(ant_tours) 
            # Optional: Apply max-min pheromone limits (tau_max, tau_min) for MMAS
            # self.pheromone_matrix = np.clip(self.pheromone_matrix, tau_min, tau_max)
            # np.fill_diagonal(self.pheromone_matrix, 0)


            self.convergence_history.append(self.best_global_distance)
            self.iteration_count_for_solve = iteration + 1

            if (iteration + 1) % 20 == 0: # Log progress less frequently
                 print(f"Iter {iteration+1}/{self.max_iterations}, Algo: {algorithm_name}, Problem: {problem_name}, Best Dist: {self.best_global_distance:.2f}")

            if self._check_early_stopping(iteration):
                print(f"早停于第{iteration + 1}轮迭代 for {algorithm_name} on {problem_name}")
                break
        
        execution_time = time.time() - start_time
        
        return ACOResult(
            best_path=self.best_global_tour,
            best_distance=self.best_global_distance,
            convergence_history=self.convergence_history,
            iteration_count=self.iteration_count_for_solve,
            execution_time=execution_time,
            problem_name=problem_name,
            algorithm_name=algorithm_name
        )

    def _check_early_stopping(self, current_iteration: int, patience: int = 20, min_delta: float = 1e-5) -> bool:
        """检查早停条件: 如果连续 'patience' 次迭代最优解没有显著改善"""
        if len(self.convergence_history) < patience + 1: 
            return False
        
        # Check if the best score has not improved by at least min_delta over the patience window
        # Compare current best (self.convergence_history[-1]) with best 'patience' iterations ago (self.convergence_history[-(patience+1)])
        if self.convergence_history[-1] >= self.convergence_history[-(patience + 1)] - min_delta:
            # Optional: A stricter check could be that *all* values in the window are very close
            # recent_improvements = np.diff(self.convergence_history[-patience:])
            # if np.all(np.abs(recent_improvements) < min_delta):
            # print(f"Early stopping triggered at iteration {current_iteration+1}: No significant improvement for {patience} iterations.")
            return True
        return False


class StandardACO(BaseACO):
    """标准蚁群算法 (Ant System - AS)"""
    def _deposit_pheromones(self, ant_tours: List[Tuple[List[int], float]]):
        """AS的信息素沉积策略 (所有蚂蚁都沉积信息素)"""
        for tour, distance in ant_tours:
            if distance == float('inf') or distance == 0: continue 
            pheromone_delta = 1.0 / distance  # Q/L, where Q=1 

            for i in range(len(tour)):
                from_city = tour[i]
                to_city = tour[(i + 1) % len(tour)]
                self.pheromone_matrix[from_city][to_city] += pheromone_delta
                if self.tsp.is_symmetric():
                    self.pheromone_matrix[to_city][from_city] += pheromone_delta


class ElitistAntSystem(BaseACO):
    """精英蚁群系统 (EAS)"""
    def __init__(
        self,
        tsp_problem: BaseTSP,
        n_ants: int = 20,
        alpha: float = 1.0,
        beta: float = 2.0,
        evaporation_rate: float = 0.1,
        initial_pheromone: float = 0.1,
        max_iterations: int = 100,
        q0: float = 0.0,
        seed: Optional[int] = None,
        elite_weight: float = 1.0, 
    ):
        super().__init__(
            tsp_problem, n_ants, alpha, beta, evaporation_rate,
            initial_pheromone, max_iterations, q0, seed
        )
        self.elite_weight = elite_weight # Weight for the global best ant's pheromone

    def _deposit_pheromones(self, ant_tours: List[Tuple[List[int], float]]):
        """EAS的信息素沉积策略: 所有蚂蚁 + 精英蚂蚁额外增强"""
        # 1. Standard pheromone deposit by all ants in the current iteration
        for tour, distance in ant_tours:
            if distance == float('inf') or distance == 0: continue
            pheromone_delta_iter = 1.0 / distance 
            for i in range(len(tour)):
                from_city = tour[i]
                to_city = tour[(i + 1) % len(tour)]
                self.pheromone_matrix[from_city][to_city] += pheromone_delta_iter
                if self.tsp.is_symmetric():
                    self.pheromone_matrix[to_city][from_city] += pheromone_delta_iter

        # 2. Elite ant (global best tour found so far) deposits additional pheromone
        if self.best_global_tour is not None and self.best_global_distance != float('inf') and self.best_global_distance > 0:
            # The amount of pheromone is e * (Q/L_gb). If Q=1, then e * (1/L_gb)
            # This is an *additional* deposit on top of evaporation.
            elite_pheromone_amount = self.elite_weight * (1.0 / self.best_global_distance)
            
            for i in range(len(self.best_global_tour)):
                from_city = self.best_global_tour[i]
                to_city = self.best_global_tour[(i + 1) % len(self.best_global_tour)]
                self.pheromone_matrix[from_city][to_city] += elite_pheromone_amount
                if self.tsp.is_symmetric():
                    self.pheromone_matrix[to_city][from_city] += elite_pheromone_amount


class ACOVisualizer:
    """ACO算法结果可视化类"""

    @staticmethod
    def plot_convergence_comparison(
        results: Dict[str, ACOResult], # Keyed by a unique run identifier
        title: str = "Compare Algorithm",
        save_path: Optional[str] = None,
        show_plot: bool = True
    ):
        """绘制多个算法的收敛曲线比较"""
        plt.figure(figsize=(12, 8))
        for run_id, result in results.items():
            label = f"{result.algorithm_name} on {result.problem_name} (Best: {result.best_distance:.2f}, Iter: {result.iteration_count})"
            plt.plot(
                result.convergence_history,
                label=label,
                linewidth=2, marker="o", markersize=3, alpha=0.8,
            )
        plt.xlabel("Iteration", fontsize=12)
        plt.ylabel("Best Path Distance", fontsize=12)
        plt.title(title, fontsize=14, fontweight="bold")
        if not results: # If results dict is empty
            plt.text(0.5, 0.5, "No data to plot.", horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
        else:
            plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        fig = plt.gcf() # Get current figure

        if save_path:
            try:
                os.makedirs(os.path.dirname(save_path), exist_ok=True) # Ensure dir exists
                fig.savefig(save_path)
                print(f"收敛曲线图已保存到: {save_path}")
            except Exception as e:
                print(f"错误: 无法保存收敛曲线图到 {save_path}. 原因: {e}")
        if show_plot:
            plt.show()
        plt.close(fig) # Close plot to free memory

    @staticmethod
    def plot_tour_visualization(
        tsp_problem: BaseTSP,
        tour: Optional[List[int]],
        title: str = "Best Path",
        save_path: Optional[str] = None,
        show_plot: bool = True
    ):
        """可视化TSP路径（仅适用于有坐标的对称TSP）"""
        if tour is None or not tour: # Check if tour is None or empty
            # print("无法可视化路径：路径未找到或为空。")
            return
        if not hasattr(tsp_problem, "coordinates") or tsp_problem.coordinates is None:
            # print("无法可视化路径：TSP问题没有坐标信息。")
            return

        coords = tsp_problem.coordinates
        fig = plt.figure(figsize=(10, 8)) 
        plt.scatter(coords[:, 0], coords[:, 1], c="red", s=100, zorder=5, label="Cities")
        for i, (x, y) in enumerate(coords):
            plt.annotate(str(i), (x, y), xytext=(5, 5), textcoords="offset points", fontsize=10, fontweight="bold")

        # Complete the tour path by connecting the last city back to the first
        tour_to_plot = tour + [tour[0]]
        tour_coords = coords[tour_to_plot] 
        plt.plot(tour_coords[:, 0], tour_coords[:, 1], "b-", linewidth=2, alpha=0.7, label="Path")
        
        start_city_coord = coords[tour[0]]
        plt.scatter(start_city_coord[0], start_city_coord[1], c="green", s=200, marker="*", zorder=6, label="Start City")

        distance = tsp_problem.calculate_tour_distance(tour) # Original tour, not tour_to_plot
        plt.title(f"{title}\nPath Distance: {distance:.2f}", fontsize=14, fontweight="bold")
        plt.xlabel("X Coordinate", fontsize=12)
        plt.ylabel("Y Coordinate", fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_path:
            try:
                os.makedirs(os.path.dirname(save_path), exist_ok=True) # Ensure dir exists
                fig.savefig(save_path)
                print(f"路径图已保存到: {save_path}")
            except Exception as e:
                print(f"错误: 无法保存路径图到 {save_path}. 原因: {e}")

        if show_plot:
            plt.show()
        plt.close(fig) 

    @staticmethod
    def create_performance_comparison_table(results: Dict[str, ACOResult], save_path: Optional[str] = None):
        """创建性能比较表格并选择性保存到文件"""
        table_lines = []
        title_line = "\n" + "=" * 35 + " 算法性能比较 " + "=" * 35
        table_lines.append(title_line)
        
        header = f"{'描述':<60} {'算法':<15} {'问题':<25} {'最优距离':<12} {'迭代次数':<10} {'执行时间(s)':<15} {'收敛速度':<10}"
        table_lines.append(header)
        table_lines.append("-" * (len(header)+20))

        if not results:
            table_lines.append("没有可供比较的结果。")
        else:
            for run_id, result in results.items(): # run_id is the unique key like "StandardACO_SymmetricTSP_timestamp"
                algo_name = result.algorithm_name
                problem_name = result.problem_name
                
                convergence_speed_str = "N/A"
                if result.convergence_history and len(result.convergence_history) > 1:
                    initial_dist = result.convergence_history[0]
                    final_dist = result.best_distance
                    if initial_dist > final_dist and initial_dist != float('inf'): 
                        target_dist = initial_dist - 0.5 * (initial_dist - final_dist)
                        convergence_iter = result.iteration_count 
                        for i, dist_val in enumerate(result.convergence_history):
                            if dist_val <= target_dist:
                                convergence_iter = i + 1
                                break
                        convergence_speed_str = f"{convergence_iter}轮"
                
                row = (f"{run_id:<62} {algo_name:<17} {problem_name:<27} {result.best_distance:<16.2f} "
                       f"{result.iteration_count:<14} {result.execution_time:<19.3f} {convergence_speed_str:<10}")
                table_lines.append(row)
        
        table_lines.append("=" * (len(header) + 20))
        
        output_text = "\n".join(table_lines)
        print(output_text)

        if save_path:
            try:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                with open(save_path, 'w', encoding='utf-8') as f:
                    f.write(output_text)
                print(f"性能比较表已保存到: {save_path}")
            except IOError as e:
                print(f"错误: 无法保存性能表到 {save_path}. 原因: {e}")


def run_single_experiment(
    tsp_problem: BaseTSP,
    problem_label: str, # e.g. "SymmetricTSP_20cities"
    aco_class, 
    aco_params: dict,
    aco_label: str, # e.g. "StandardACO"
    run_id_prefix: str, # Timestamp or global run ID
    output_dir: str,
    show_plots: bool
) -> Tuple[str, ACOResult]:
    """运行单个ACO实验并返回唯一ID和结果"""
    full_run_id = f"{run_id_prefix}_{problem_label}_{aco_label}"
    print(f"\n开始实验: {full_run_id}")
    
    # Pass tsp_problem explicitly, others from aco_params
    solver_instance = aco_class(tsp_problem=tsp_problem, **aco_params)
    
    result = solver_instance.solve(problem_name=problem_label, algorithm_name=aco_label)

    print(f"实验 {full_run_id} - 完成. 最优距离: {result.best_distance:.2f}, 耗时: {result.execution_time:.2f}s, 迭代: {result.iteration_count}")

    if result.best_path: # Only plot if a path was found
        if hasattr(tsp_problem, "coordinates") and tsp_problem.coordinates is not None:
            tour_plot_filename = f"{full_run_id}_best_tour.png"
            tour_plot_path = os.path.join(output_dir, tour_plot_filename)
            ACOVisualizer.plot_tour_visualization(
                tsp_problem, result.best_path,
                title=f"{aco_label} on {problem_label} - Best Path",
                save_path=tour_plot_path,
                show_plot=show_plots
            )
    return full_run_id, result


def main_experiment_controller(args):
    """根据命令行参数组织和运行实验"""
    
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"结果将保存到目录: {args.output_dir}")

    if args.seed is not None:
        np.random.seed(args.seed)
        random.seed(args.seed)
        print(f"全局随机种子已设置为: {args.seed}")

    all_experiment_results: Dict[str, ACOResult] = {}
    # Unique prefix for all files from this execution
    execution_timestamp_prefix = time.strftime("%Y%m%d-%H%M%S") 

    # Common ACO parameters from args
    common_aco_args = {
        'n_ants': args.n_ants,
        'alpha': args.alpha,
        'beta': args.beta,
        'evaporation_rate': args.evaporation_rate,
        'initial_pheromone': args.initial_pheromone,
        'max_iterations': args.max_iterations,
        'q0': args.q0,
        'seed': args.seed # Pass seed to individual ACO solvers
    }
    
    elitist_specific_args = {'elite_weight': args.elite_weight}

    # --- Symmetric TSP Experiments ---
    if args.tsp_type in ["symmetric", "both"]:
        print("\n" + "="*25 + " 对称TSP实验 " + "="*25)
        sym_tsp_instance = SymmetricTSP.create_random_instance(
            n_cities=args.n_cities_sym,
            coordinate_range=((args.coord_min, args.coord_max), (args.coord_min, args.coord_max)),
            seed=args.seed 
        )
        problem_desc_sym = f"SymmetricTSP-{args.n_cities_sym}cities"
        print(f"问题: {problem_desc_sym}")

        sym_run_results = {} # For this TSP type's convergence plot

        if args.aco_variant in ["standard", "both"]:
            run_id, res = run_single_experiment(
                tsp_problem=sym_tsp_instance, problem_label=problem_desc_sym,
                aco_class=StandardACO, aco_params=common_aco_args, aco_label="StandardACO",
                run_id_prefix=execution_timestamp_prefix, output_dir=args.output_dir, show_plots=args.visualize
            )
            all_experiment_results[run_id] = res
            sym_run_results[run_id] = res


        if args.aco_variant in ["elitist", "both"]:
            eas_params = {**common_aco_args, **elitist_specific_args}
            run_id, res = run_single_experiment(
                tsp_problem=sym_tsp_instance, problem_label=problem_desc_sym,
                aco_class=ElitistAntSystem, aco_params=eas_params, aco_label="ElitistAS",
                run_id_prefix=execution_timestamp_prefix, output_dir=args.output_dir, show_plots=args.visualize
            )
            all_experiment_results[run_id] = res
            sym_run_results[run_id] = res
        
        if sym_run_results:
            conv_plot_path = os.path.join(args.output_dir, f"{execution_timestamp_prefix}_{problem_desc_sym}_convergence.png")
            ACOVisualizer.plot_convergence_comparison(
                sym_run_results, title=f"{problem_desc_sym} - Compare",
                save_path=conv_plot_path, show_plot=args.visualize
            )

    # --- Asymmetric TSP Experiments ---
    if args.tsp_type in ["asymmetric", "both"]:
        print("\n" + "="*25 + " 非对称TSP实验 " + "="*25)
        asym_tsp_instance = AsymmetricTSP.create_random_instance(
            n_cities=args.n_cities_asym,
            asymmetry_factor=args.asymmetry_factor,
            seed=args.seed
        )
        problem_desc_asym = f"AsymmetricTSP-{args.n_cities_asym}cities"
        print(f"问题: {problem_desc_asym} (非对称因子: {args.asymmetry_factor})")
        
        asym_run_results = {}

        if args.aco_variant in ["standard", "both"]:
            run_id, res = run_single_experiment(
                tsp_problem=asym_tsp_instance, problem_label=problem_desc_asym,
                aco_class=StandardACO, aco_params=common_aco_args, aco_label="StandardACO",
                run_id_prefix=execution_timestamp_prefix, output_dir=args.output_dir, show_plots=args.visualize
            )
            all_experiment_results[run_id] = res
            asym_run_results[run_id] = res

        if args.aco_variant in ["elitist", "both"]:
            eas_params = {**common_aco_args, **elitist_specific_args}
            run_id, res = run_single_experiment(
                tsp_problem=asym_tsp_instance, problem_label=problem_desc_asym,
                aco_class=ElitistAntSystem, aco_params=eas_params, aco_label="ElitistAS",
                run_id_prefix=execution_timestamp_prefix, output_dir=args.output_dir, show_plots=args.visualize
            )
            all_experiment_results[run_id] = res
            asym_run_results[run_id] = res

        if asym_run_results:
            conv_plot_path = os.path.join(args.output_dir, f"{execution_timestamp_prefix}_{problem_desc_asym}_convergence.png")
            ACOVisualizer.plot_convergence_comparison(
                asym_run_results, title=f"{problem_desc_asym} - Compare",
                save_path=conv_plot_path, show_plot=args.visualize
            )

    # --- Overall Performance Summary Table ---
    if all_experiment_results:
        summary_table_path = os.path.join(args.output_dir, f"{execution_timestamp_prefix}_all_performance_summary.txt")
        ACOVisualizer.create_performance_comparison_table(
            all_experiment_results,
            save_path=summary_table_path
        )
    else:
        print("没有实验被配置运行，无法生成性能总结表。")

    print("\n所有已配置的实验均已完成！")
    print(f"详细结果、图表和性能总结已保存到目录: {args.output_dir}")


if __name__ == "__main__":
    cli_parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                         description="运行蚁群优化 (ACO) 算法解决旅行商问题 (TSP)。")

    # General settings
    general_group = cli_parser.add_argument_group('常规设置')
    general_group.add_argument('--output_dir', type=str, default="aco_results", help="保存所有输出文件 (图片, 结果文本) 的目录。")
    general_group.add_argument('--seed', type=int, default=None, help="用于numpy和random的全局随机种子，以确保可复现性。如果为None，则不设置特定种子。")
    general_group.add_argument('--visualize', action='store_true', help="运行时显示可视化图片 (例如收敛曲线, 最优路径图)。默认不显示。")
    cli_parser.set_defaults(visualize=False)

    # TSP problem settings
    tsp_group = cli_parser.add_argument_group('TSP 问题设置')
    tsp_group.add_argument('--tsp_type', type=str, choices=["symmetric", "asymmetric", "both"], default="both", help="要运行的TSP问题类型。'both' 将同时运行对称和非对称问题。")
    tsp_group.add_argument('--n_cities_sym', type=int, default=20, help="对称TSP (Symmetric TSP) 的城市数量。")
    tsp_group.add_argument('--coord_min', type=float, default=0.0, help="对称TSP城市坐标的最小值。")
    tsp_group.add_argument('--coord_max', type=float, default=100.0, help="对称TSP城市坐标的最大值。")
    tsp_group.add_argument('--n_cities_asym', type=int, default=20, help="非对称TSP (Asymmetric TSP) 的城市数量。")
    tsp_group.add_argument('--asymmetry_factor', type=float, default=0.4, help="非对称TSP的非对称因子。值越大，非对称性越强。")

    # ACO algorithm settings
    aco_group = cli_parser.add_argument_group('ACO 算法设置')
    aco_group.add_argument('--aco_variant', type=str, choices=["standard", "elitist", "both"], default="both", help="要运行的ACO算法变体。'both' 将同时运行标准ACO和精英蚁群系统。")
    aco_group.add_argument('--n_ants', type=int, default=20, help="每次迭代中使用的蚂蚁数量。")
    aco_group.add_argument('--max_iterations', type=int, default=100, help="算法的最大迭代次数。")
    aco_group.add_argument('--alpha', type=float, default=1.0, help="信息素重要性因子 (α)。控制信息素轨迹的相对重要性。")
    aco_group.add_argument('--beta', type=float, default=2.0, help="启发式信息重要性因子 (β)。控制启发式信息 (如距离的倒数) 的相对重要性。")
    aco_group.add_argument('--evaporation_rate', type=float, default=0.1, help="信息素蒸发率 (ρ)。值在 (0, 1] 之间，表示信息素的衰减速度。")
    aco_group.add_argument('--initial_pheromone', type=float, default=0.1, help="路径上初始信息素的浓度 (τ₀)。")
    aco_group.add_argument('--q0', type=float, default=0.0, 
                           help="Ant Colony System (ACS) 中的探索/利用参数。 "
                                "值为0时，进行随机比例选择 (如标准ACO/EAS)。"
                                "值 > 0 时 (例如0.9)，蚂蚁有 q0 的概率选择最优的下一城市 (利用)，有 1-q0 的概率进行随机比例选择 (探索)。")
    
    # Elitist Ant System specific settings
    eas_group = cli_parser.add_argument_group('精英蚁群系统 (EAS) 特定设置')
    eas_group.add_argument('--elite_weight', type=float, default=2.0, help="精英蚁群系统中精英解 (全局最优解) 对信息素更新的额外权重 (e)。")
    
    parsed_args = cli_parser.parse_args()

    main_experiment_controller(parsed_args)
