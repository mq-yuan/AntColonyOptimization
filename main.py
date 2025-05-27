import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional
import random
from dataclasses import dataclass
import time


@dataclass
class ACOResult:
    """ACO算法结果数据类"""

    best_path: Optional[List[int]]
    best_distance: float
    convergence_history: List[float]
    iteration_count: int
    execution_time: float


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
        if np.any(np.diag(self.distance_matrix) != 0):
            raise ValueError("城市到自身的距离应为0")

    @abstractmethod
    def is_symmetric(self) -> bool:
        """判断是否为对称TSP"""
        pass

    def calculate_tour_distance(self, tour: List[int]) -> float:
        """计算旅行路径的总距离"""
        if len(tour) != self.n_cities:
            raise ValueError(f"路径长度应为{self.n_cities}")

        total_distance = 0.0
        for i in range(len(tour)):
            current_city = tour[i]
            next_city = tour[(i + 1) % len(tour)]
            total_distance += self.distance_matrix[current_city][next_city]

        return total_distance

    def get_distance(self, from_city: int, to_city: int) -> float:
        """获取两城市间距离"""
        return self.distance_matrix[from_city][to_city]

    def generate_random_cities(
        self, n_cities: int, coordinate_range: Tuple[float, float] = (0, 100)
    ) -> np.ndarray:
        """生成随机城市坐标"""
        np.random.seed(42)  # 确保可重现性
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
        elif city_coordinates is not None:
            distance_matrix = self._calculate_euclidean_distances(city_coordinates)
            super().__init__(distance_matrix, city_names)
            self.coordinates = city_coordinates
        else:
            raise ValueError("必须提供距离矩阵或城市坐标")

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
        cls, n_cities: int, coordinate_range: Tuple[float, float] = (0, 100)
    ):
        """创建随机对称TSP实例"""
        np.random.seed(42)
        coordinates = np.random.uniform(
            coordinate_range[0], coordinate_range[1], (n_cities, 2)
        )
        return cls(city_coordinates=coordinates)


class AsymmetricTSP(BaseTSP):
    """非对称TSP问题类"""

    def __init__(
        self, distance_matrix: np.ndarray, city_names: Optional[List[str]] = None
    ):
        super().__init__(distance_matrix, city_names)

    def is_symmetric(self) -> bool:
        return False

    @classmethod
    def create_random_instance(cls, n_cities: int, asymmetry_factor: float = 0.3):
        """创建随机非对称TSP实例"""
        np.random.seed(42)
        # 先生成对称矩阵
        coordinates = np.random.uniform(0, 100, (n_cities, 2))
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
    ):
        self.tsp = tsp_problem
        self.n_ants = n_ants
        self.alpha = alpha  # 信息素重要性
        self.beta = beta  # 启发式信息重要性
        self.evaporation_rate = evaporation_rate  # 信息素蒸发率
        self.initial_pheromone = initial_pheromone
        self.max_iterations = max_iterations

        # 初始化信息素矩阵
        self.pheromone_matrix = np.full(
            (self.tsp.n_cities, self.tsp.n_cities), initial_pheromone, dtype=float
        )

        # 计算启发式信息矩阵 (1/distance)
        self.heuristic_matrix = np.zeros((self.tsp.n_cities, self.tsp.n_cities))
        for i in range(self.tsp.n_cities):
            for j in range(self.tsp.n_cities):
                if i != j:
                    self.heuristic_matrix[i][j] = 1.0 / self.tsp.get_distance(i, j)

        # 记录最优解
        self.best_global_tour = None
        self.best_global_distance = float("inf")
        self.convergence_history = []

    def _construct_ant_solution(self, ant_id: int) -> Tuple[List[int], float]:
        """构造单只蚂蚁的解"""
        # 随机选择起始城市
        current_city = random.randint(0, self.tsp.n_cities - 1)
        tour = [current_city]
        unvisited = set(range(self.tsp.n_cities)) - {current_city}

        while unvisited:
            next_city = self._select_next_city(current_city, unvisited)
            tour.append(next_city)
            unvisited.remove(next_city)
            current_city = next_city

        distance = self.tsp.calculate_tour_distance(tour)
        return tour, distance

    def _select_next_city(self, current_city: int, unvisited: set) -> int:
        """选择下一个城市"""
        probabilities = []
        cities = list(unvisited)

        for city in cities:
            pheromone = self.pheromone_matrix[current_city][city] ** self.alpha
            heuristic = self.heuristic_matrix[current_city][city] ** self.beta
            probabilities.append(pheromone * heuristic)

        # 归一化概率
        total_prob = sum(probabilities)
        if total_prob == 0:
            return random.choice(cities)

        probabilities = [p / total_prob for p in probabilities]

        # 轮盘赌选择
        r = random.random()
        cumulative_prob = 0
        for i, prob in enumerate(probabilities):
            cumulative_prob += prob
            if r <= cumulative_prob:
                return cities[i]

        return cities[-1]  # fallback

    def _evaporate_pheromones(self):
        """信息素蒸发"""
        self.pheromone_matrix *= 1 - self.evaporation_rate

    @abstractmethod
    def _deposit_pheromones(self, ant_tours: List[Tuple[List[int], float]]):
        """信息素沉积 - 由子类实现"""
        pass

    def solve(self) -> ACOResult:
        """求解TSP问题"""
        start_time = time.time()

        for iteration in range(self.max_iterations):
            # 构造所有蚂蚁的解
            ant_tours = []
            for ant in range(self.n_ants):
                tour, distance = self._construct_ant_solution(ant)
                ant_tours.append((tour, distance))

                # 更新全局最优解
                if distance < self.best_global_distance:
                    self.best_global_distance = distance
                    self.best_global_tour = tour.copy()

            # 信息素更新
            self._evaporate_pheromones()
            self._deposit_pheromones(ant_tours)

            # 记录收敛历史
            self.convergence_history.append(self.best_global_distance)

            # 早停条件检查
            if len(self.convergence_history) > 20:
                recent_improvements = np.diff(self.convergence_history[-20:])
                if np.all(recent_improvements >= -1e-6):  # 几乎没有改进
                    print(f"早停于第{iteration + 1}轮迭代")
                    break

        execution_time = time.time() - start_time

        return ACOResult(
            best_path=self.best_global_tour,
            best_distance=self.best_global_distance,
            convergence_history=self.convergence_history,
            iteration_count=len(self.convergence_history),
            execution_time=execution_time,
        )


class StandardACO(BaseACO):
    """标准蚁群算法"""

    def _deposit_pheromones(self, ant_tours: List[Tuple[List[int], float]]):
        """标准ACO的信息素沉积策略"""
        for tour, distance in ant_tours:
            pheromone_delta = 1.0 / distance  # Q/L的简化形式，Q=1

            for i in range(len(tour)):
                from_city = tour[i]
                to_city = tour[(i + 1) % len(tour)]
                self.pheromone_matrix[from_city][to_city] += pheromone_delta

                # 对于对称TSP，同时更新反向边
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
        elite_weight: float = 1.0,
    ):
        super().__init__(
            tsp_problem,
            n_ants,
            alpha,
            beta,
            evaporation_rate,
            initial_pheromone,
            max_iterations,
        )
        self.elite_weight = elite_weight  # 精英解的权重

    def _deposit_pheromones(self, ant_tours: List[Tuple[List[int], float]]):
        """EAS的信息素沉积策略"""
        # 标准蚂蚁的信息素沉积
        for tour, distance in ant_tours:
            pheromone_delta = 1.0 / distance

            for i in range(len(tour)):
                from_city = tour[i]
                to_city = tour[(i + 1) % len(tour)]
                self.pheromone_matrix[from_city][to_city] += pheromone_delta

                if self.tsp.is_symmetric():
                    self.pheromone_matrix[to_city][from_city] += pheromone_delta

        # 精英解的额外信息素沉积
        if self.best_global_tour is not None:
            elite_pheromone_delta = self.elite_weight / self.best_global_distance

            for i in range(len(self.best_global_tour)):
                from_city = self.best_global_tour[i]
                to_city = self.best_global_tour[(i + 1) % len(self.best_global_tour)]
                self.pheromone_matrix[from_city][to_city] += elite_pheromone_delta

                if self.tsp.is_symmetric():
                    self.pheromone_matrix[to_city][from_city] += elite_pheromone_delta


class ACOVisualizer:
    """ACO算法结果可视化类"""

    @staticmethod
    def plot_convergence_comparison(results: dict, title: str = "算法收敛性比较"):
        """绘制多个算法的收敛曲线比较"""
        plt.figure(figsize=(12, 8))

        for label, result in results.items():
            plt.plot(
                result.convergence_history,
                label=f"{label} (Best: {result.best_distance:.2f})",
                linewidth=2,
                marker="o",
                markersize=3,
                alpha=0.8,
            )

        plt.xlabel("Iteration", fontsize=12)
        plt.ylabel("Best Path Distance", fontsize=12)
        plt.title(title, fontsize=14, fontweight="bold")
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_tour_visualization(
        tsp_problem: BaseTSP, tour: List[int], title: str = "最优路径"
    ):
        """可视化TSP路径（仅适用于有坐标的对称TSP）"""
        if not hasattr(tsp_problem, "coordinates"):
            print("无法可视化路径：TSP问题没有坐标信息")
            return

        coords = tsp_problem.coordinates
        plt.figure(figsize=(10, 8))

        # 绘制城市
        plt.scatter(coords[:, 0], coords[:, 1], c="red", s=100, zorder=5)

        # 标注城市编号
        for i, (x, y) in enumerate(coords):
            plt.annotate(
                str(i),
                (x, y),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=10,
                fontweight="bold",
            )

        # 绘制路径
        tour_coords = coords[tour + [tour[0]]]  # 回到起点
        plt.plot(tour_coords[:, 0], tour_coords[:, 1], "b-", linewidth=2, alpha=0.7)

        # 标出起点
        start_city = tour[0]
        plt.scatter(
            coords[start_city, 0],
            coords[start_city, 1],
            c="green",
            s=200,
            marker="*",
            zorder=6,
            label="start",
        )

        distance = tsp_problem.calculate_tour_distance(tour)
        plt.title(f"{title}\nPath Distance: {distance:.2f}", fontsize=14, fontweight="bold")
        plt.xlabel("X", fontsize=12)
        plt.ylabel("Y", fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def create_performance_comparison_table(results: dict):
        """创建性能比较表格"""
        print("\n" + "=" * 80)
        print("算法性能比较")
        print("=" * 80)
        print(
            f"{'算法':<15} {'最优距离':<12} {'迭代次数':<10} {'执行时间(s)':<12} {'收敛速度':<10}"
        )
        print("-" * 80)

        for label, result in results.items():
            # 计算收敛速度（前50%改进所需迭代数）
            if len(result.convergence_history) > 1:
                initial_distance = result.convergence_history[0]
                final_distance = result.best_distance
                target_distance = initial_distance - 0.5 * (
                    initial_distance - final_distance
                )

                convergence_iteration = len(result.convergence_history)
                for i, distance in enumerate(result.convergence_history):
                    if distance <= target_distance:
                        convergence_iteration = i + 1
                        break
                convergence_speed = f"{convergence_iteration}轮"
            else:
                convergence_speed = "N/A"

            print(
                f"{label:<17} {result.best_distance:<17.2f} {result.iteration_count:<14} "
                f"{result.execution_time:<15.3f} {convergence_speed:<10}"
            )
        print("=" * 80)


def run_comprehensive_experiments():
    """运行综合实验"""
    print("开始蚁群算法综合实验...")
    print("\n" + "=" * 50)

    # 实验1: 对称TSP问题比较
    print("实验1: 对称TSP问题 - ACO vs EAS")
    print("=" * 50)

    # 创建对称TSP实例
    symmetric_tsp = SymmetricTSP.create_random_instance(n_cities=20)
    print(f"问题规模: {symmetric_tsp.n_cities}个城市")

    # 运行标准ACO
    print("运行标准ACO...")
    aco_standard = StandardACO(symmetric_tsp, n_ants=20, max_iterations=100)
    result_aco_sym = aco_standard.solve()

    # 运行精英蚁群系统
    print("运行精英蚁群系统...")
    eas = ElitistAntSystem(
        symmetric_tsp, n_ants=20, max_iterations=100, elite_weight=2.0
    )
    result_eas_sym = eas.solve()

    # 实验2: 非对称TSP问题比较
    print("\n实验2: 非对称TSP问题 - ACO vs EAS")
    print("=" * 50)

    # 创建非对称TSP实例
    asymmetric_tsp = AsymmetricTSP.create_random_instance(
        n_cities=20, asymmetry_factor=0.4
    )
    print(f"问题规模: {asymmetric_tsp.n_cities}个城市 (非对称)")

    # 运行标准ACO
    print("运行标准ACO...")
    aco_asym = StandardACO(asymmetric_tsp, n_ants=20, max_iterations=100)
    result_aco_asym = aco_asym.solve()

    # 运行精英蚁群系统
    print("运行精英蚁群系统...")
    eas_asym = ElitistAntSystem(
        asymmetric_tsp, n_ants=20, max_iterations=100, elite_weight=2.0
    )
    result_eas_asym = eas_asym.solve()

    # 结果分析和可视化
    print("\n分析结果...")

    # 对称TSP结果
    symmetric_results = {"ACO": result_aco_sym, "EAS": result_eas_sym}

    # 非对称TSP结果
    asymmetric_results = {"ACO": result_aco_asym, "EAS": result_eas_asym}

    # 显示性能比较表格
    print("\n对称TSP性能比较:")
    ACOVisualizer.create_performance_comparison_table(symmetric_results)

    print("\n非对称TSP性能比较:")
    ACOVisualizer.create_performance_comparison_table(asymmetric_results)

    # 绘制收敛曲线
    ACOVisualizer.plot_convergence_comparison(
        symmetric_results, "TSP: ACO vs EAS"
    )
    ACOVisualizer.plot_convergence_comparison(
        asymmetric_results, "ATSP: ACO vs EAS "
    )

    # 可视化最优路径（仅对称TSP有坐标）
    ACOVisualizer.plot_tour_visualization(
        symmetric_tsp, result_eas_sym.best_path, "TSP Best Path (EAS)"
    )

    # 算法特性分析
    print("\n算法特性分析:")
    print("=" * 50)

    # 比较对称vs非对称TSP的算法表现
    print(f"对称TSP - ACO最优解: {result_aco_sym.best_distance:.2f}")
    print(f"对称TSP - EAS最优解: {result_eas_sym.best_distance:.2f}")
    print(f"非对称TSP - ACO最优解: {result_aco_asym.best_distance:.2f}")
    print(f"非对称TSP - EAS最优解: {result_eas_asym.best_distance:.2f}")

    # 分析EAS的改进效果
    sym_improvement = (
        (result_aco_sym.best_distance - result_eas_sym.best_distance)
        / result_aco_sym.best_distance
    ) * 100
    asym_improvement = (
        (result_aco_asym.best_distance - result_eas_asym.best_distance)
        / result_aco_asym.best_distance
    ) * 100

    print("\nEAS相比标准ACO的改进:")
    print(f"对称TSP: {sym_improvement:.2f}%")
    print(f"非对称TSP: {asym_improvement:.2f}%")

    return {
        "symmetric_results": symmetric_results,
        "asymmetric_results": asymmetric_results,
        "symmetric_tsp": symmetric_tsp,
        "asymmetric_tsp": asymmetric_tsp,
    }


if __name__ == "__main__":
    # 设置随机种子以确保可重现性
    np.random.seed(42)
    random.seed(42)

    # 运行综合实验
    experiment_results = run_comprehensive_experiments()

    print("\n实验完成！")
    print("=" * 50)
    print("主要发现:")
    print("1. 精英蚁群系统(EAS)通常比标准ACO收敛更快")
    print("2. EAS在利用已发现的好解方面更有效")
    print("3. 非对称TSP比对称TSP更具挑战性")
    print("4. 两种算法都能有效处理对称和非对称TSP问题")
