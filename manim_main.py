from manim import *
import numpy as np
import random

# 确保 main.py 在同一目录或 Python 路径中
from main import SymmetricTSP, StandardACO

# --- Manim 可视化配置 (基于用户提供的manim_main.py) ---
seed = 3407  #
NUM_CITIES = 5  # 城市数量
N_ANTS_ACO = 2  # ACO 算法中使用的蚂蚁数量
VIS_ITERATIONS = 5  # 在 Manim 中完整可视化的 ACO 迭代次数
# 用户希望每个蚂蚁在子图中显示，因此 ANTS_TO_VISUALIZE_PER_ITER 设置为 N_ANTS_ACO
ANTS_TO_VISUALIZE_PER_ITER = N_ANTS_ACO  #

# 颜色和样式
CITY_COLOR = BLUE_D  # 城市颜色
CITY_RADIUS = 0.38  # 城市节点半径
CITY_LABEL_SCALE = 0.8  # 城市标签缩放
ANT_COLOR = YELLOW_D  # 蚂蚁移动点的颜色
ANT_PATH_COLOR = WHITE  # 蚂蚁构建路径时的颜色
ANT_TEMP_PATH_STROKE = 3.5  # 蚂蚁临时路径的线宽

# 信息素可视化增强
PHEROMONE_LOW_COLOR = BLUE_E  # 信息素较低时的颜色
PHEROMONE_HIGH_COLOR = BLUE_E # 信息素较高时的颜色
PHEROMONE_MIN_OPACITY = 0.15  # 增加最小不透明度，使弱路径更可见
PHEROMONE_MAX_OPACITY = 1.0
PHEROMONE_MIN_STROKE = 1.5  # 增加最小线宽
PHEROMONE_MAX_STROKE = 18.0  # 略微减小最大线宽以平衡视觉
PHEROMONE_VISUAL_EXPONENT = 1.5  # >1 强调强信息素路径, <1 使弱路径也较明显

BEST_PATH_COLOR = WHITE  # 最优路径颜色
BEST_PATH_STROKE = 7.0  # 最优路径线宽，略作调整以突出
TEXT_COLOR_ON_LIGHT_BG = WHITE # 在浅色背景上使用的文本颜色

SUBPLOT_SCALE_FACTOR = 0.30  # 每个子图的缩放因子
SUBPLOT_BUFFER = 0.6  # 子图之间的间距

class ACOVisualizerManim(MovingCameraScene):
    """
    使用 Manim 可视化蚁群算法解决 TSP 问题的场景。
    """
    def setup_aco_problem(self):
        """初始化 TSP 问题和 ACO 求解器。"""
        self.tsp_problem = SymmetricTSP.create_random_instance(
            n_cities=NUM_CITIES,
            seed=seed,
            coordinate_range=((-2.0, 2.0), (-3.0, 1.0))  # 城市坐标范围，Y轴调整以适应屏幕
        )
        self.city_coords_3d = [np.array([c[0], c[1], 0]) for c in self.tsp_problem.coordinates]

        self.aco_solver = StandardACO(
            tsp_problem=self.tsp_problem,
            n_ants=N_ANTS_ACO,
            alpha=1.0,
            beta=2.0,
            evaporation_rate=0.25, # 较高的蒸发率以便更快看到视觉变化
            initial_pheromone=1.0, # 初始信息素值
            max_iterations=VIS_ITERATIONS, # Manim迭代次数即为ACO迭代次数
            q0=0.0, # 标准 ACO (AS) 的行为
            seed=seed
        )
        # 获取初始最大信息素值，用于后续可视化标准化
        self.current_max_pheromone = np.max(self.aco_solver.pheromone_matrix[np.isfinite(self.aco_solver.pheromone_matrix)])
        if self.current_max_pheromone == 0: # 避免除以零
            self.current_max_pheromone = 1.0

    # 在 ACOVisualizerManim 类中
    def get_pheromone_visual_props(self, pheromone_value: float, max_pheromone_for_norm: float) -> tuple[float, float, ManimColor]:
        """
        根据信息素值和用于标准化的最大信息素值，计算线条的透明度、粗细和颜色。
        """
        current_max_phero_safe = max_pheromone_for_norm if max_pheromone_for_norm > 1e-9 else 1.0 # 使用传入的最大值
        normalized_pheromone = np.clip(pheromone_value / current_max_phero_safe, 0, 1)
        
        # 应用指数增强差异
        enhanced_normalized_pheromone = normalized_pheromone ** PHEROMONE_VISUAL_EXPONENT

        opacity = PHEROMONE_MIN_OPACITY + enhanced_normalized_pheromone * (PHEROMONE_MAX_OPACITY - PHEROMONE_MIN_OPACITY)
        stroke_width = PHEROMONE_MIN_STROKE + enhanced_normalized_pheromone * (PHEROMONE_MAX_STROKE - PHEROMONE_MIN_STROKE)
        color = interpolate_color(PHEROMONE_LOW_COLOR, PHEROMONE_HIGH_COLOR, enhanced_normalized_pheromone)
        return opacity, stroke_width, color

    def construct(self):
        """构建 Manim 场景和动画。"""
        self.setup_aco_problem()
        self.camera.background_color = GREY_E # 使用浅灰色背景，以便黑色路径可见

        # --- 标题和信息文本 ---
        title = Text("ACO for TSP", font_size=36, color=TEXT_COLOR_ON_LIGHT_BG).to_edge(UP)
        iter_text = Text(f"Iteration: 0/{VIS_ITERATIONS}", font_size=24, color=TEXT_COLOR_ON_LIGHT_BG).to_corner(UL)
        dist_text = Text(f"Current Best Distance: N/A", font_size=24, color=TEXT_COLOR_ON_LIGHT_BG).next_to(iter_text, DOWN, aligned_edge=LEFT)
        self.add(title, iter_text, dist_text) #尽早添加，确保它们存在

        # --- 绘制主要城市布局 (用于信息素和最终路径显示) ---
        # self.main_city_nodes_with_labels 将存储主场景中的城市对象（点+标签）
        self.main_city_nodes_with_labels = VGroup()
        for i, coord in enumerate(self.city_coords_3d):
            dot = Dot(coord, radius=CITY_RADIUS, color=CITY_COLOR, z_index=2) # 城市点z_index较高
            label_text = Text(str(i), font_size=18, color=TEXT_COLOR_ON_LIGHT_BG).move_to(dot.get_center()).scale(CITY_LABEL_SCALE).set_z_index(3)
            city_node_group = VGroup(dot, label_text)
            self.main_city_nodes_with_labels.add(city_node_group)

        self.play(LaggedStart(*[Create(city_obj) for city_obj in self.main_city_nodes_with_labels], lag_ratio=0.1), run_time=1.5)

        # --- 初始化主要信息素路径 ---
        pheromone_lines_mobjects = {} # 键: (城市i, 城市j), 值: Line mobject
        initial_pheromone_vgroup = VGroup().set_z_index(1) # 信息素路径在城市点之下
        for i in range(NUM_CITIES):
            for j in range(i + 1, NUM_CITIES): # 对称TSP
                opacity, stroke, p_color = self.get_pheromone_visual_props(self.aco_solver.pheromone_matrix[i, j], self.current_max_pheromone)
                main_dot_i = self.main_city_nodes_with_labels[i].submobjects[0] # 城市i的点
                main_dot_j = self.main_city_nodes_with_labels[j].submobjects[0] # 城市j的点
                line = Line(
                    main_dot_i.get_center(), main_dot_j.get_center(),
                    stroke_color=p_color, stroke_opacity=opacity, stroke_width=stroke
                )
                pheromone_lines_mobjects[(i,j)] = line
                initial_pheromone_vgroup.add(line)
        self.add(initial_pheromone_vgroup)

        current_best_path_mobject = VGroup().set_z_index(5) # 最优路径在最前

        # --- ACO 迭代可视化主循环 ---
        for iter_k in range(VIS_ITERATIONS):
            self.play(iter_text.animate.become(Text(f"Iteration: {iter_k+1}/{VIS_ITERATIONS}", font_size=24, color=TEXT_COLOR_ON_LIGHT_BG).to_corner(UL)))
            self.play(FadeOut(current_best_path_mobject, scale=0.5), run_time=0.05) # 快速淡出旧的最优路径

            iteration_ant_tours = []
            
            # --- 设置子图显示 ---
            subplot_main_vgroup = VGroup()
            ant_specific_city_groups_in_subplots = [] # 存储每个子图中城市对象的引用

            if ANTS_TO_VISUALIZE_PER_ITER > 0:
                for ant_vis_idx in range(ANTS_TO_VISUALIZE_PER_ITER):
                    # 为每个子图创建主城市节点（点和标签）的深拷贝
                    subplot_cities_with_labels = VGroup(*[node.copy() for node in self.main_city_nodes_with_labels])
                    ant_specific_city_groups_in_subplots.append(subplot_cities_with_labels)

                    subplot_title_text = Text(f"Ant {ant_vis_idx + 1} Path", font_size=20, color=TEXT_COLOR_ON_LIGHT_BG)
                    
                    single_subplot_content = VGroup(subplot_title_text, subplot_cities_with_labels)
                    single_subplot_content.scale(SUBPLOT_SCALE_FACTOR) # 缩放整个子图内容
                    
                    # 将标题放置在缩放后的城市组上方
                    subplot_title_text.next_to(subplot_cities_with_labels, UP, buff=0.15 * SUBPLOT_SCALE_FACTOR) # 调整buff以适应缩放
                    
                    subplot_main_vgroup.add(single_subplot_content)

                # 在网格中排列子图
                cols = ANTS_TO_VISUALIZE_PER_ITER # 一行显示所有子图
                rows = 1
                if ANTS_TO_VISUALIZE_PER_ITER > 2: # 如果子图过多，可以考虑换行
                    cols = int(np.ceil(np.sqrt(ANTS_TO_VISUALIZE_PER_ITER)))
                    rows = int(np.ceil(ANTS_TO_VISUALIZE_PER_ITER / cols))
                
                subplot_main_vgroup.arrange_in_grid(rows=rows, cols=cols, buff=SUBPLOT_BUFFER)
                # 将子图组移动到屏幕特定区域，例如下方，并调整大小
                subplot_main_vgroup.move_to(DOWN * 1.5).scale_to_fit_width(self.camera.frame_width * 0.7)

                # 动画：暂时隐藏主视图元素，显示子图组
                self.play(
                    FadeOut(self.main_city_nodes_with_labels, initial_pheromone_vgroup, current_best_path_mobject, scale=0.8),
                    FadeIn(subplot_main_vgroup, scale=1.0), # 子图从中心放大出现
                    run_time=0.7
                )

            # --- 1. ACO: 蚂蚁构建路径。在子图中可视化。 ---
            all_ants_subplot_animations = [] # 存储所有蚂蚁在子图中的动画序列

            for ant_idx in range(self.aco_solver.n_ants):
                tour, distance = self.aco_solver._construct_ant_solution(ant_idx)
                if tour and len(tour) == self.tsp_problem.n_cities:
                    iteration_ant_tours.append((tour, distance))
                    if distance < self.aco_solver.best_global_distance:
                        self.aco_solver.best_global_distance = distance
                        self.aco_solver.best_global_tour = tour[:]

                if ant_idx < ANTS_TO_VISUALIZE_PER_ITER and tour:
                    # 获取当前蚂蚁对应的子图中的城市对象组
                    current_subplot_cities = ant_specific_city_groups_in_subplots[ant_idx]
                    
                    ant_dot_subplot = Dot(
                        current_subplot_cities[tour[0]].submobjects[0].get_center(), # 起始于子图中第一个城市
                        color=ANT_COLOR, radius=0.09 * SUBPLOT_SCALE_FACTOR * 1.5 # 适当调整蚂蚁点的大小
                    ).set_z_index(10) # 确保蚂蚁点在最前
                    
                    current_ant_path_anim_sequence = [FadeIn(ant_dot_subplot)] # 动画序列以显示蚂蚁开始
                    temp_path_segments_this_ant_subplot = VGroup() # 存储此蚂蚁在子图中的路径段

                    for city_path_idx in range(len(tour)):
                        start_node_dot_subplot = current_subplot_cities[tour[city_path_idx]].submobjects[0] # 子图中的起始点
                        end_node_original_idx = tour[(city_path_idx + 1) % len(tour)]
                        end_node_dot_subplot = current_subplot_cities[end_node_original_idx].submobjects[0] # 子图中的结束点

                        segment = Line(
                            start_node_dot_subplot.get_center(), end_node_dot_subplot.get_center(),
                            stroke_color=ANT_PATH_COLOR,
                            stroke_width=ANT_TEMP_PATH_STROKE * SUBPLOT_SCALE_FACTOR, # 缩放路径线宽
                            stroke_opacity=0.9, z_index=5 # 路径在蚂蚁点之下
                        )
                        temp_path_segments_this_ant_subplot.add(segment)
                        
                        current_ant_path_anim_sequence.append(
                            AnimationGroup(
                                ant_dot_subplot.animate.move_to(end_node_dot_subplot.get_center()),
                                Create(segment),
                                rate_func=linear,
                                run_time=(2.0 / len(tour)) # 调整动画时间以获得良好观感
                            )
                        )
                    # 添加此蚂蚁的点和路径的淡出动画
                    current_ant_path_anim_sequence.append(FadeOut(temp_path_segments_this_ant_subplot, scale=0.5, run_time=0.3))
                    current_ant_path_anim_sequence.append(FadeOut(ant_dot_subplot, scale=0.5, run_time=0.3))
                    all_ants_subplot_animations.append(Succession(*current_ant_path_anim_sequence, lag_ratio=1.0)) # 每个蚂蚁的动画作为一个Succession

            # 播放所有蚂蚁在各自子图中的路径构建动画（可选择并行或串行）
            if all_ants_subplot_animations:
                self.play(*all_ants_subplot_animations) # 并行播放所有蚂蚁动画
                # for ant_anim_sequence in all_ants_subplot_animations: # 串行播放，一个接一个
                #     self.play(ant_anim_sequence)

            # --- 从子图过渡回主视图 ---
            if ANTS_TO_VISUALIZE_PER_ITER > 0 and subplot_main_vgroup.submobjects: # 检查是否有子图被显示
                 self.play(
                    FadeOut(subplot_main_vgroup, scale=0.9), # 子图缩小消失
                    FadeIn(self.main_city_nodes_with_labels, initial_pheromone_vgroup, scale=1.0), # 恢复主视图元素
                    run_time=0.7
                )

            # ======== 开始修改信息素动画 ========
            # --- 2. ACO: 执行信息素蒸发 (求解器内部逻辑) ---
            self.aco_solver._evaporate_pheromones()

            # --- 动画演示蒸发效果 ---
            # 此刻, self.aco_solver.pheromone_matrix 中的值是仅蒸发后的。
            # 我们需要基于这些值来计算视觉属性。
            max_phero_after_evap_stage = np.max(self.aco_solver.pheromone_matrix[np.isfinite(self.aco_solver.pheromone_matrix)])
            if max_phero_after_evap_stage <= 1e-9:  # 避免除以零或负数
                max_phero_after_evap_stage = 1e-9 # 如果所有信息素都蒸发完了，用一个极小值

            evaporation_anims_list = []
            evaporation_stage_title = Text("Pheromone Evaporation", font_size=18, color=TEXT_COLOR_ON_LIGHT_BG).to_corner(DR)
            self.play(Write(evaporation_stage_title), run_time=0.3)

            for (i,j), line_mobj in pheromone_lines_mobjects.items():
                # 从已蒸发的矩阵中获取信息素值
                avg_phero_after_evap = (self.aco_solver.pheromone_matrix[i, j] + self.aco_solver.pheromone_matrix[j, i]) / 2.0
                # 使用仅蒸发后的最大信息素进行归一化
                new_opacity, new_stroke, new_color = self.get_pheromone_visual_props(avg_phero_after_evap, max_phero_after_evap_stage)
                evaporation_anims_list.append(
                    line_mobj.animate.set_stroke(opacity=new_opacity, width=new_stroke).set_color(new_color)
                )
            if evaporation_anims_list:
                self.play(AnimationGroup(*evaporation_anims_list, lag_ratio=0.0, run_time=0.6))
            
            self.play(FadeOut(evaporation_stage_title), run_time=0.3)

            # 存储仅蒸发后，沉积前的信息素状态，用于判断哪些路径真正得到了增强
            phero_matrix_after_evap_before_deposit = self.aco_solver.pheromone_matrix.copy()

            # --- 3. ACO: 执行信息素沉积 (求解器内部逻辑) ---
            if iteration_ant_tours: # iteration_ant_tours 是从蚂蚁路径构建阶段收集的
                self.aco_solver._deposit_pheromones(iteration_ant_tours)
            
            # 此刻, self.aco_solver.pheromone_matrix 是本轮迭代的最终信息素状态。
            # 更新类属性 self.current_max_pheromone，它将作为下一次调用 get_pheromone_visual_props 的默认最大值参考
            final_max_phero_this_iteration = np.max(self.aco_solver.pheromone_matrix[np.isfinite(self.aco_solver.pheromone_matrix)])
            if final_max_phero_this_iteration > 1e-9:
                self.current_max_pheromone = final_max_phero_this_iteration
            else:
                self.current_max_pheromone = 1e-9 # Fallback

            # --- 动画演示沉积效果 ---
            deposition_anims_list = []
            deposition_highlight_anims_list = [] # 用于高亮显示信息素显著增加的路径
            
            deposition_stage_title = Text("Pheromone Deposition", font_size=18, color=TEXT_COLOR_ON_LIGHT_BG).to_corner(DR)
            self.play(Write(deposition_stage_title), run_time=0.3)

            for (i,j), line_mobj in pheromone_lines_mobjects.items():
                final_avg_phero = (self.aco_solver.pheromone_matrix[i, j] + self.aco_solver.pheromone_matrix[j, i]) / 2.0
                # 使用本轮迭代最终的最大信息素 (self.current_max_pheromone) 进行归一化
                new_opacity, new_stroke, new_color = self.get_pheromone_visual_props(final_avg_phero, self.current_max_pheromone)
                
                deposition_anims_list.append(
                    line_mobj.animate.set_stroke(opacity=new_opacity, width=new_stroke).set_color(new_color)
                )

                # 检查这条边上的信息素是否因为沉积而真正增加
                phero_on_edge_before_deposit = (phero_matrix_after_evap_before_deposit[i,j] + phero_matrix_after_evap_before_deposit[j,i]) / 2.0
                # if final_avg_phero > phero_on_edge_before_deposit + 1e-6: # 如果信息素确实增加了
                #     # 为这些路径添加一个 "脉冲" 或 "闪烁" 效果
                #     # 这里使用rate_func=there_and_back来实现短暂的视觉增强
                #     glow_color = YELLOW # 使用一个明亮的颜色作为闪烁色
                #     deposition_highlight_anims_list.append(
                #         line_mobj.animate(rate_func=there_and_back, run_time=0.5) # there_and_back 使其变化后恢复
                #                  .set_stroke(width=new_stroke + 2.0, color=glow_color) # 短暂增粗并改变颜色
                #                  # Manim 会自动处理恢复到 deposition_anims_list 中设定的最终状态（颜色和线宽）
                #                  # 如果不自动恢复，则需要再加一步 .set_stroke(width=new_stroke, color=new_color)
                #     )
            
            # 先播放所有路径到它们沉积后最终状态的动画
            if deposition_anims_list:
                self.play(AnimationGroup(*deposition_anims_list, lag_ratio=0.0, run_time=0.8))
            
            # # 然后再播放高亮动画，突出显示那些信息素增加的路径
            # if deposition_highlight_anims_list:
            #     self.play(AnimationGroup(*deposition_highlight_anims_list, lag_ratio=0.05, run_time=0.5))
            
            self.play(FadeOut(deposition_stage_title), run_time=0.3)

            # ======== 结束修改信息素动画 ========

            # --- 5. 更新并动画显示主视图的最优路径 ---
            # (这部分代码保持不变)
            if self.aco_solver.best_global_tour:
                # current_best_path_mobject 已经在迭代开始时FadeOut，这里直接创建新的
                current_best_path_mobject = VGroup().set_z_index(20) # 确保最优路径在最前
                best_tour_node_indices = self.aco_solver.best_global_tour
                for k_path_idx in range(len(best_tour_node_indices)):
                    u_node_idx = best_tour_node_indices[k_path_idx]
                    v_node_idx = best_tour_node_indices[(k_path_idx + 1) % len(best_tour_node_indices)]
                    
                    dot_u_main_view = self.main_city_nodes_with_labels[u_node_idx].submobjects[0] # 主视图中的U点
                    dot_v_main_view = self.main_city_nodes_with_labels[v_node_idx].submobjects[0] # 主视图中的V点
                    
                    path_line = Line(dot_u_main_view.get_center(), dot_v_main_view.get_center(), color=BEST_PATH_COLOR, stroke_width=BEST_PATH_STROKE)
                    current_best_path_mobject.add(path_line)
                self.play(Create(current_best_path_mobject), run_time=0.5)
            
            # 更新最优距离文本
            current_dist_val = self.aco_solver.best_global_distance
            current_dist_str = f"{current_dist_val:.2f}" if current_dist_val != float('inf') else "N/A"
            self.play(dist_text.animate.become(Text(f"Current Best Distance: {current_dist_str}", font_size=24, color=TEXT_COLOR_ON_LIGHT_BG).next_to(iter_text, DOWN, aligned_edge=LEFT)))
            self.wait(0.3) # 每轮迭代结束时稍作停顿

        # --- 动画结束 ---
        final_overall_dist_val = self.aco_solver.best_global_distance
        final_overall_dist_str = f"{final_overall_dist_val:.2f}" if final_overall_dist_val != float('inf') else "未找到"
        final_summary_message = Text(f"Final! Best Distance: {final_overall_dist_str}", font_size=28, color=TEXT_COLOR_ON_LIGHT_BG).next_to(title, DOWN, buff=0.5)
        self.play(Write(final_summary_message))
        if current_best_path_mobject.submobjects: # 检查是否有最优路径对象
            self.play(Indicate(current_best_path_mobject, color=ANT_COLOR, scale_factor=1.1, repetitions=3), run_time=2)

        self.wait(3) # 保持最后一帧几秒钟

