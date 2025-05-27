from manim import *
import numpy as np
import random

# 确保 main.py 在同一目录或 Python 路径中
from main import SymmetricTSP, StandardACO

seed=3407
# --- Manim 可视化配置 ---
NUM_CITIES = 5                                  # 城市数量 (建议保持较小以便清晰显示)
N_ANTS_ACO = 2                                 # ACO 算法中使用的蚂蚁数量
VIS_ITERATIONS = 5                              # 在 Manim 中完整可视化的 ACO 迭代次数
ANTS_TO_VISUALIZE_PER_ITER = 2                  # 每次迭代中详细显示路径的蚂蚁数量

# 颜色和样式
CITY_COLOR = BLUE_D                             # 城市颜色
CITY_RADIUS = 0.38                              # 城市节点半径
CITY_LABEL_SCALE = 0.8                          # 城市标签缩放
ANT_COLOR = YELLOW_D                            # 蚂蚁移动点的颜色
ANT_PATH_COLOR = BLACK                       # 蚂蚁构建路径时的颜色
ANT_TEMP_PATH_STROKE = 3.5                      # 蚂蚁临时路径的线宽
PHEROMONE_COLOR = GRAY_B                        # 信息素路径颜色
PHEROMONE_MIN_OPACITY = 0.05                    # 信息素最小不透明度
PHEROMONE_MAX_OPACITY = 0.9                     # 信息素最大不透明度
PHEROMONE_MIN_STROKE = 0.5                      # 信息素最小线宽
PHEROMONE_MAX_STROKE = 10.0                      # 信息素最大线宽
BEST_PATH_COLOR = GREEN_D                       # 最优路径颜色
BEST_PATH_STROKE = 9.0                          # 最优路径线宽
TEXT_COLOR = WHITE                              # 文本颜色

class ACOVisualizerManim(MovingCameraScene):
    """
    使用 Manim 可视化蚁群算法解决 TSP 问题的场景。
    """
    def setup_aco_problem(self):
        """初始化 TSP 问题和 ACO 求解器。"""
        # 使用 main.py 中的 SymmetricTSP 创建随机 TSP 实例
        # Manim 的坐标系通常 y 轴向上，x 轴向右。
        # 为了在默认相机视图中良好显示，坐标范围设置得较小。
        self.tsp_problem = SymmetricTSP.create_random_instance(
            n_cities=NUM_CITIES,
            seed=seed,  # 使用固定种子以保证可复现性
            coordinate_range=((-4.0, 4.0), (-3.5, 2.5)) # 城市坐标范围
        )
        # 将二维坐标转换为三维，z=0
        self.city_coords_3d = [np.array([c[0], c[1], 0]) for c in self.tsp_problem.coordinates]

        # 初始化 StandardACO 求解器
        self.aco_solver = StandardACO(
            tsp_problem=self.tsp_problem,
            n_ants=N_ANTS_ACO,
            alpha=1.0,
            beta=2.0,
            evaporation_rate=0.25,  # 较高的蒸发率以便更快看到视觉变化
            initial_pheromone=1.0,  # 初始信息素值
            max_iterations=VIS_ITERATIONS, # Manim迭代次数即为ACO迭代次数
            q0=0.0, # 标准 ACO (AS) 的行为
            seed=seed
        )
        # 获取初始最大信息素值，用于后续可视化标准化
        self.current_max_pheromone = np.max(self.aco_solver.pheromone_matrix[np.isfinite(self.aco_solver.pheromone_matrix)])
        if self.current_max_pheromone == 0: # 避免除以零
            self.current_max_pheromone = 1.0

    def get_pheromone_visual_props(self, pheromone_value: float) -> tuple[float, float]:
        """根据信息素值计算线条的透明度和粗细。"""
        # 将信息素值标准化到 0-1 范围
        normalized_pheromone = np.clip(pheromone_value / (self.current_max_pheromone + 1e-6), 0, 1)
        opacity = PHEROMONE_MIN_OPACITY + normalized_pheromone * (PHEROMONE_MAX_OPACITY - PHEROMONE_MIN_OPACITY)
        stroke_width = PHEROMONE_MIN_STROKE + normalized_pheromone * (PHEROMONE_MAX_STROKE - PHEROMONE_MIN_STROKE)
        return opacity, stroke_width

    def construct(self):
        """构建 Manim 场景和动画。"""
        self.setup_aco_problem()

        # --- 标题和信息文本 ---
        title = Text("ACO for TSP", font_size=36).to_edge(UP)
        iter_text = Text(f"Iteration: 0/{VIS_ITERATIONS}", font_size=24, color=TEXT_COLOR).to_corner(UL)
        dist_text = Text(f"Current Best Distance: N/A", font_size=24, color=TEXT_COLOR).next_to(iter_text, DOWN, aligned_edge=LEFT)
        self.play(Write(title))
        self.add(iter_text, dist_text)

        # --- 绘制城市 ---
        city_mobjects_group = VGroup() # 用于容纳城市点和标签
        city_dots_for_camera = VGroup() # 仅含点，用于相机对焦
        for i, coord in enumerate(self.city_coords_3d):
            dot = Dot(coord, radius=CITY_RADIUS, color=CITY_COLOR)
            label = Text(str(i), font_size=18, color=TEXT_COLOR).move_to(dot.get_center()).scale(CITY_LABEL_SCALE)
            city_mobjects_group.add(VGroup(dot, label)) # 将点和标签组合
            city_dots_for_camera.add(dot)

        self.play(LaggedStart(*[Create(city_obj) for city_obj in city_mobjects_group], lag_ratio=0.1), run_time=1.5)

        # # 调整相机视角以适应城市布局
        # self.camera.frame.scale_to_fit_width(city_dots_for_camera.get_width() * 1.8)
        # self.camera.frame.move_to(city_dots_for_camera.get_center())

        # --- 初始化信息素路径 (Manim 对象) ---
        # 使用字典存储 Line 对象，键为 (城市i, 城市j)
        pheromone_lines_mobjects = {}
        initial_pheromone_vgroup = VGroup() # 用于初次显示所有信息素路径
        for i in range(NUM_CITIES):
            for j in range(i + 1, NUM_CITIES): # 对称TSP，仅需上半部分
                opacity, stroke = self.get_pheromone_visual_props(self.aco_solver.pheromone_matrix[i, j])
                line = Line(
                    self.city_coords_3d[i], self.city_coords_3d[j],
                    stroke_color=PHEROMONE_COLOR, stroke_opacity=opacity, stroke_width=stroke
                )
                pheromone_lines_mobjects[(i,j)] = line
                initial_pheromone_vgroup.add(line)
        self.add_foreground_mobjects(initial_pheromone_vgroup) #确保信息素在城市点之上，但在蚂蚁和最优路径之下

        current_best_path_mobject = VGroup() # 用于显示当前全局最优路径

        # --- ACO 迭代可视化主循环 ---
        for iter_k in range(VIS_ITERATIONS):
            self.play(iter_text.animate.become(Text(f"Iteration: {iter_k+1}/{VIS_ITERATIONS}", font_size=24, color=TEXT_COLOR).to_corner(UL)))

            # 1. 执行ACO的一轮迭代：蚂蚁构建路径
            iteration_ant_tours = [] # 存储本轮所有蚂蚁的 (路径, 距离)
            temp_ant_paths_to_fade = VGroup() # 临时存储本轮要显示的蚂蚁路径动画对象

            self.play(FadeOut(current_best_path_mobject, scale=0.5), run_time=0.1)
            for ant_idx in range(self.aco_solver.n_ants):
                # 调用ACO内部方法构建单只蚂蚁的解
                tour, distance = self.aco_solver._construct_ant_solution(ant_idx)
                if tour and len(tour) == self.tsp_problem.n_cities: # 确保是有效路径
                    iteration_ant_tours.append((tour, distance))
                    # 更新全局最优解 (逻辑来自 BaseACO.solve)
                    if distance < self.aco_solver.best_global_distance:
                        self.aco_solver.best_global_distance = distance
                        self.aco_solver.best_global_tour = tour[:] # 复制列表

                # 可视化部分蚂蚁的路径构建过程
                if ant_idx < ANTS_TO_VISUALIZE_PER_ITER and tour:
                    ant_dot_visual = Dot(self.city_coords_3d[tour[0]], color=ANT_COLOR, radius=0.09).set_z_index(10)
                    self.add(ant_dot_visual)
                    
                    current_ant_path_segments = VGroup()
                    path_segment_animations = []
                    for city_path_idx in range(len(tour)):
                        start_node_coord = self.city_coords_3d[tour[city_path_idx]]
                        end_node_idx = tour[(city_path_idx + 1) % len(tour)]
                        end_node_coord = self.city_coords_3d[end_node_idx]
                        
                        segment = Line(start_node_coord, end_node_coord, stroke_color=ANT_PATH_COLOR, stroke_width=ANT_TEMP_PATH_STROKE, stroke_opacity=0.7).set_z_index(5)
                        current_ant_path_segments.add(segment)
                        
                        # 蚂蚁移动和路径创建动画
                        path_segment_animations.append(
                            AnimationGroup(
                                ant_dot_visual.animate.move_to(end_node_coord),
                                Create(segment),
                                rate_func=linear, # 匀速
                                run_time = 3.5 / len(tour) # 每段路径的动画时间
                            )
                        )
                    
                    if path_segment_animations:
                        self.play(Succession(*path_segment_animations, lag_ratio=1.0)) # 播放单只蚂蚁的完整路径动画
                    
                    temp_ant_paths_to_fade.add(current_ant_path_segments)
                    self.remove(ant_dot_visual) # 移除蚂蚁点

            # 淡出本轮所有蚂蚁的临时路径
            if temp_ant_paths_to_fade:
                self.play(FadeOut(temp_ant_paths_to_fade), run_time=0.5)

            # 2. 信息素蒸发
            self.aco_solver._evaporate_pheromones()

            # 3. 信息素沉积 (使用本轮收集到的蚂蚁路径)
            if iteration_ant_tours: # 确保有有效路径进行信息素沉积
                self.aco_solver._deposit_pheromones(iteration_ant_tours)
            
            # 更新当前最大信息素值，用于可视化标准化
            new_max_phero = np.max(self.aco_solver.pheromone_matrix[np.isfinite(self.aco_solver.pheromone_matrix)])
            if new_max_phero > 0 : self.current_max_pheromone = new_max_phero


            # 4. 更新信息素路径的 Manim 对象动画
            pheromone_update_animations = []
            for (i,j), line_mobject in pheromone_lines_mobjects.items():
                # 获取对称路径上的信息素 (因为ACO内部可能非对称更新，但可视化常为对称)
                phero_val_ij = self.aco_solver.pheromone_matrix[i, j]
                phero_val_ji = self.aco_solver.pheromone_matrix[j, i]
                avg_phero = (phero_val_ij + phero_val_ji) / 2.0 # 对称显示时取平均

                new_opacity, new_stroke = self.get_pheromone_visual_props(avg_phero)
                pheromone_update_animations.append(
                    line_mobject.animate.set_stroke(opacity=new_opacity, width=new_stroke)
                )
            if pheromone_update_animations:
                self.play(*pheromone_update_animations, run_time=0.7)
            
            # 5. 更新并动画显示当前最优路径
            if self.aco_solver.best_global_tour:
                # 先移除旧的最优路径（如果存在）
                current_best_path_mobject = VGroup().set_z_index(20) # 确保最优路径在最前
                best_tour_nodes = self.aco_solver.best_global_tour
                for k in range(len(best_tour_nodes)):
                    u, v = best_tour_nodes[k], best_tour_nodes[(k + 1) % len(best_tour_nodes)]
                    line = Line(self.city_coords_3d[u], self.city_coords_3d[v], color=BEST_PATH_COLOR, stroke_width=BEST_PATH_STROKE)
                    current_best_path_mobject.add(line)
                self.play(Create(current_best_path_mobject), run_time=0.4)
            
            # 更新最优距离文本
            dist_val = self.aco_solver.best_global_distance
            dist_str = f"{dist_val:.2f}" if dist_val != float('inf') else "N/A"
            self.play(dist_text.animate.become(Text(f"Current Best Distance: {dist_str}", font_size=24, color=TEXT_COLOR).next_to(iter_text, DOWN, aligned_edge=LEFT)))
            self.wait(0.2) # 每轮迭代结束时稍作停顿

        # --- 动画结束 ---
        final_dist_val = self.aco_solver.best_global_distance
        final_dist_str = f"{final_dist_val:.2f}" if final_dist_val != float('inf') else "未找到"
        final_message = Text(f"Final! Best Distance: {final_dist_str}", font_size=28, color=TEXT_COLOR).next_to(title, DOWN, buff=0.5)
        self.play(Write(final_message))
        if current_best_path_mobject: # 高亮最终最优路径
            self.play(Indicate(current_best_path_mobject, color=YELLOW, scale_factor=1.1, repetitions=3), run_time=2)

        self.wait(3) # 保持最后一帧几秒钟

# --- 如何运行 ---
# 1. 确保你已经安装了 Manim Community version (pip install manim) 及其依赖。
# 2. 将此代码保存为 `manim.py` 文件，并与你的 `main.py` 文件放在同一目录下。
# 3. 打开终端或命令行，导航到该目录。
# 4. 运行 Manim 命令: manim -pql manim.py ACOVisualizerManim
#    - `-pql` 表示低质量预览并播放。你可以使用 `-pqm` (中等质量) 或 `-pqh` (高质量)。
#    - Manim 会在 `media` 子目录中生成视频文件。

