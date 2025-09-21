import pulp
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import gurobipy # 雖然 PuLP 會自動調用，但明確導入可以確認環境中存在此模組
import os # 導入 os 模組來處理文件和目錄

# =============================================================================
# --- 參數配置區 ---
# 您只需要修改這裡的數值即可
# =============================================================================
# 1. 設定目標節點的數量
NUM_NODES = 25  # <--- 在這裡修改節點數量

# 2. 設定固定的通訊範圍 (半徑)
FIXED_RADIUS = 200 # <--- 在這裡修改所有節點的通訊半徑

# 3. 設定場地大小
AREA_SIZE = 3000

# 4. 設定求解器最長運行時間 (秒)
SOLVER_TIME_LIMIT = 18000 # 1分鐘

# 5. 【新增】設定自動運行的次數
NUM_RUNS = 30 # <--- 設定要自動運行幾次

# 6. 【新增】設定儲存結果圖片的資料夾名稱
OUTPUT_DIR = "results"
# =============================================================================


# --- 1. 數據生成與幾何預處理 (無變動) ---

def generate_data(num_nodes, area_size, radius):
    """
    根據配置生成模擬數據。
    新增約束：確保任意兩個圓的圓心距離不小於半徑，防止一個圓完全包含另一個。
    """
    # np.random.seed(42) # 已註解掉，確保每次運行生成不同的隨機場景
    depot = np.array([area_size / 2, area_size / 2])
    
    valid_coords = []
    max_attempts_per_node = 1000
    
    for i in range(num_nodes):
        for attempt in range(max_attempts_per_node):
            new_coord = np.random.rand(2) * area_size
            is_valid = True
            for existing_coord in valid_coords:
                distance = np.linalg.norm(new_coord - existing_coord)
                if distance < radius:
                    is_valid = False
                    break
            if is_valid:
                valid_coords.append(new_coord)
                break
        else:
            raise Exception(f"無法在 {max_attempts_per_node} 次嘗試內為第 {i+1} 個節點找到有效位置。")

    nodes_coords = np.array(valid_coords)
    nodes_radii = np.full(num_nodes, radius)
    return depot, nodes_coords, nodes_radii

def get_circle_intersections(p1, r1, p2, r2):
    d = np.linalg.norm(p1 - p2)
    if d > r1 + r2 or d < abs(r1 - r2) or d == 0: return []
    a = (r1**2 - r2**2 + d**2) / (2 * d)
    h = math.sqrt(max(0, r1**2 - a**2))
    p_mid = p1 + a * (p2 - p1) / d
    x_mid, y_mid = p_mid[0], p_mid[1]
    x2, y2 = p2[0] - p1[0], p2[1] - p1[1]
    i1 = [x_mid + h * y2 / d, y_mid - h * x2 / d]
    i2 = [x_mid - h * y2 / d, y_mid + h * x2 / d]
    if d == r1 + r2: return [np.array(i1)]
    return [np.array(i1), np.array(i2)]

def build_candidate_points(depot, nodes_coords, nodes_radii):
    candidate_points = [depot] + list(nodes_coords)
    num_nodes = len(nodes_coords)
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            intersections = get_circle_intersections(nodes_coords[i], nodes_radii[i], nodes_coords[j], nodes_radii[j])
            candidate_points.extend(intersections)
    return np.array(candidate_points)


# --- 2. 視覺化函數 (已修改) ---
def visualize_solution(depot, nodes_coords, nodes_radii, candidate_points, path_indices, total_distance, area_size, filename):
    """
    【修改】: 
    1. 新增 'filename' 參數。
    2. 將 plt.show() 改為 plt.savefig() 來儲存圖片。
    3. 新增 plt.close() 關閉圖形，避免在迴圈中消耗過多記憶體。
    """
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.set_aspect('equal', adjustable='box')
    
    for i in range(len(nodes_coords)):
        circle = patches.Circle(nodes_coords[i], nodes_radii[i], color='blue', alpha=0.1, label='_nolegend_')
        ax.add_patch(circle)
        ax.plot(nodes_coords[i][0], nodes_coords[i][1], 'b.', markersize=5)

    ax.plot(depot[0], depot[1], 'r*', markersize=20, label='Depot (Start/End)')
    path_coords = candidate_points[path_indices]
    
    for i in range(len(path_coords) - 1):
        start_point = path_coords[i]
        end_point = path_coords[i+1]
        ax.arrow(start_point[0], start_point[1], end_point[0] - start_point[0], end_point[1] - start_point[1],
                 head_width=area_size*0.015, head_length=area_size*0.02, fc='black', ec='black', length_includes_head=True)

    ax.plot(path_coords[:, 0], path_coords[:, 1], 'go', markersize=7, label='Optimal Stops')

    ax.set_xlim(0, area_size)
    ax.set_ylim(0, area_size)
    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")
    ax.set_title(f"Optimal Drone Path (Solver: Gurobi)\nTotal Distance: {total_distance:.2f}", fontsize=16)
    ax.legend()
    ax.grid(True)
    
    # --- 主要修改處 ---
    plt.savefig(filename, bbox_inches='tight', dpi=150) # 儲存圖片，dpi可調整解析度
    plt.close(fig) # 關閉圖形，釋放記憶體
    print(f"結果已儲存至: {filename}")


# --- 3. 主程序：建模與求解 (已修改) ---
def main():
    # 【新增】: 檢查並建立儲存結果的資料夾
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"已建立資料夾: {OUTPUT_DIR}")

    # 【新增】: 主循環，運行 NUM_RUNS 次
    for run in range(NUM_RUNS):
        print(f"\n{'='*25} 開始運行第 {run + 1} / {NUM_RUNS} 次 {'='*25}")
        
        try:
            DEPOT, NODES_COORDS, NODES_RADII = generate_data(num_nodes=NUM_NODES, area_size=AREA_SIZE, radius=FIXED_RADIUS)
            NUM_TARGET_NODES = len(NODES_COORDS)

            candidate_points = build_candidate_points(DEPOT, NODES_COORDS, NODES_RADII)
            NUM_CANDIDATES = len(candidate_points)
            print(f"總目標節點數: {NUM_TARGET_NODES}")
            print(f"總候選停留點數: {NUM_CANDIDATES}")

            dist_matrix = np.linalg.norm(candidate_points[:, np.newaxis, :] - candidate_points[np.newaxis, :, :], axis=2)
            covers_matrix = np.zeros((NUM_CANDIDATES, NUM_TARGET_NODES))
            for i in range(NUM_CANDIDATES):
                for k in range(NUM_TARGET_NODES):
                    if np.linalg.norm(candidate_points[i] - NODES_COORDS[k]) <= NODES_RADII[k] + 1e-6:
                        covers_matrix[i, k] = 1

            V = range(NUM_CANDIDATES)
            N = range(NUM_TARGET_NODES)
            M = NUM_CANDIDATES

            prob = pulp.LpProblem("Drone_TSPN", pulp.LpMinimize)
            x = pulp.LpVariable.dicts("x", (V, V), cat='Binary')
            y = pulp.LpVariable.dicts("y", V, cat='Binary')
            u = pulp.LpVariable.dicts("u", V, lowBound=0, upBound=M-1, cat='Continuous')

            prob += pulp.lpSum(dist_matrix[i][j] * x[i][j] for i in V for j in V if i != j), "Total_Distance"

            for k in N:
                prob += pulp.lpSum(covers_matrix[i][k] * y[i] for i in V) >= 1, f"Cover_Node_{k}"
            for i in V:
                prob += pulp.lpSum(x[i][j] for j in V if i != j) == y[i], f"Flow_Out_{i}"
                prob += pulp.lpSum(x[j][i] for j in V if i != j) == y[i], f"Flow_In_{i}"
            prob += y[0] == 1, "Start_At_Depot"
            for i in V:
                if i == 0: continue
                for j in V:
                    if i == j or j == 0: continue
                    prob += u[i] - u[j] + M * x[i][j] <= M - 1, f"Subtour_Elim_{i}_{j}"
            for i in V:
                if i == 0: continue
                prob += u[i] >= 1 * y[i], f"u_lower_bound_{i}"

            print(f"\n模型建立完成，開始使用 Gurobi 求解 (最長 {SOLVER_TIME_LIMIT} 秒)...")
            solver = pulp.GUROBI_CMD(msg=True, timeLimit=SOLVER_TIME_LIMIT)
            prob.solve(solver)

            print("\n--- 求解結果 ---")
            print("狀態:", pulp.LpStatus[prob.status])
            
            # 【修改】: 檢查狀態是否為 Optimal 或 Feasible
            if prob.status in [pulp.LpStatusOptimal, pulp.LpStatusNotSolved]: # NotSolved 有時表示找到可行解但超時
                if pulp.value(prob.objective) is None:
                    print("求解器超時，未能找到任何可行解。")
                    continue # 繼續下一次運行

                total_dist = pulp.value(prob.objective)
                print(f"路徑總長度: {total_dist:.2f}")

                path_indices = [0]
                current_node_idx = 0
                active_edges = {}
                for i in V:
                    for j in V:
                        if i != j and pulp.value(x[i][j]) == 1:
                            active_edges[i] = j
                
                if 0 in active_edges:
                    while True:
                        next_node_idx = active_edges.get(current_node_idx)
                        if next_node_idx is None or next_node_idx == 0:
                            break
                        path_indices.append(next_node_idx)
                        current_node_idx = next_node_idx
                    path_indices.append(0)
                    
                    print("路徑 (停留點索引):")
                    print(" -> ".join(map(str, path_indices)))

                    # 【修改】: 組合檔案名並調用修改後的視覺化函數
                    output_filename = os.path.join(OUTPUT_DIR, f"run_{run+1}_nodes_{NUM_NODES}.png")
                    visualize_solution(DEPOT, NODES_COORDS, NODES_RADII, candidate_points, path_indices, total_dist, AREA_SIZE, output_filename)
                else:
                    print("錯誤：求解器聲稱找到解，但無法提取有效路徑。")

            else:
                print("未能在規定時間內找到最佳解或可行解。")

        except Exception as e:
            print(f"\n第 {run + 1} 次運行出錯: {e}")

# --- 執行主程序 ---
if __name__ == "__main__":
    main()