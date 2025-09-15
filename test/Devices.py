import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import math

# --- 解決中文亂碼問題 ---
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] 
plt.rcParams['axes.unicode_minus'] = False

# --- 參數設定 ---
AREA_WIDTH = 2000
AREA_HEIGHT = 2000
NUM_NODES = 25
COMM_RADIUS = 200    # 通訊半徑 (公尺)

# --- 需求 1：節點間的最小間距 ---
MIN_NODE_DISTANCE = 100.0

# --- 基地台 (Base Station) 參數 ---
BS_COORDS = (AREA_WIDTH / 2, AREA_HEIGHT / 2)
BS_SIZE = 100
BS_COLOR = 'green'

# --- 需求 2：計算節點與基地台的最小安全距離 ---
# 為了保證節點的通訊範圍(圓)不接觸基地台(方塊)，
# 節點中心與基地台中心的最短距離，近似為 通訊半徑 + 基地台對角線長度的一半。
# 這樣計算能確保在最壞情況下 (節點朝著方塊的角) 也不會接觸。
BS_DIAGONAL = math.sqrt(2 * BS_SIZE**2)
MIN_DIST_TO_BS_CENTER = COMM_RADIUS + (BS_DIAGONAL / 2)
print(f"為確保通訊範圍不覆蓋基地台，節點中心與基地台中心的最小距離被設定為: {MIN_DIST_TO_BS_CENTER:.2f} 公尺")

# --- 使用抑制性隨機佈點演算法生成節點 ---
nodes_coords = []
max_attempts = NUM_NODES * 100

print("正在生成地面裝置座標...")

while len(nodes_coords) < NUM_NODES:
    candidate_x = np.random.uniform(0, AREA_WIDTH)
    candidate_y = np.random.uniform(0, AREA_HEIGHT)
    
    is_valid = True
    
    # 檢查 1: 候選點與基地台的距離是否足夠遠
    dist_to_bs_center = math.sqrt((candidate_x - BS_COORDS[0])**2 + (candidate_y - BS_COORDS[1])**2)
    if dist_to_bs_center < MIN_DIST_TO_BS_CENTER:
        is_valid = False

    # 檢查 2: 如果與基地台距離OK，再檢查與其他已存在節點的距離
    if is_valid:
        for node_x, node_y in nodes_coords:
            distance = math.sqrt((candidate_x - node_x)**2 + (candidate_y - node_y)**2)
            if distance < MIN_NODE_DISTANCE:
                is_valid = False
                break
            
    if is_valid:
        nodes_coords.append((candidate_x, candidate_y))

    max_attempts -= 1
    if max_attempts <= 0:
        print(f"警告：在最大嘗試次數內，只成功生成了 {len(nodes_coords)}/{NUM_NODES} 個節點。")
        break

print(f"成功生成 {len(nodes_coords)} 個裝置。")

if not nodes_coords:
    exit()
    
node_x_coords, node_y_coords = zip(*nodes_coords)

# --- 開始繪圖 ---
fig, ax = plt.subplots(figsize=(10, 10))

# 繪製基地台
bs_bottom_left = (BS_COORDS[0] - BS_SIZE / 2, BS_COORDS[1] - BS_SIZE / 2)
base_station_patch = patches.Rectangle(
    bs_bottom_left, BS_SIZE, BS_SIZE, 
    facecolor=BS_COLOR, edgecolor='black', linewidth=1.5, label='基地台 (BS)'
)
ax.add_patch(base_station_patch)


# 繪製地面裝置和其通訊範圍
ax.plot(node_x_coords, node_y_coords, 'ro', markersize=5, label='地面裝置')

for x, y in nodes_coords:
    communication_range = patches.Circle(
        (x, y), COMM_RADIUS, linestyle='--', edgecolor='blue', facecolor='blue', alpha=0.1
    )
    ax.add_patch(communication_range)

ax.plot([], [], '--', color='blue', label=f'通訊範圍 (半徑={COMM_RADIUS}m)')

# --- 美化圖表 ---
ax.set_xlim(0, AREA_WIDTH)
ax.set_ylim(0, AREA_HEIGHT)
ax.set_aspect('equal', adjustable='box')

plt.title(f'滿足多重約束的地面裝置分佈圖', fontsize=16)
plt.xlabel('X 座標 (公尺)', fontsize=12)
plt.ylabel('Y 座標 (公尺)', fontsize=12)
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(loc='upper right')

plt.show()