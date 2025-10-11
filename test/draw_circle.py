import matplotlib.pyplot as plt
import numpy as np

# --- 1. 參數設定 ---
# 圓的參數
CIRCLE_RADIUS = 5.0
CIRCLE_CENTER = np.array([0, 0])

# --- 2. 建立畫布 ---
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_aspect('equal', adjustable='box')

# --- 3. 繪製基礎圓形 ---
# a. 繪製黑色虛線圓
theta_circle = np.linspace(0, 2 * np.pi, 200)
x_circle = CIRCLE_CENTER[0] + CIRCLE_RADIUS * np.cos(theta_circle)
y_circle = CIRCLE_CENTER[1] + CIRCLE_RADIUS * np.sin(theta_circle)
ax.plot(x_circle, y_circle, 'k--', linewidth=1.5)

# b. 繪製圓心
ax.plot(CIRCLE_CENTER[0], CIRCLE_CENTER[1], 'ko', markersize=6)

# --- 4. 在圓周上選取兩個隨機點 ---
min_angle_separation = np.pi / 4
angle1 = np.random.uniform(0, 2 * np.pi)
while True:
    angle2 = np.random.uniform(0, 2 * np.pi)
    diff = np.abs(angle1 - angle2)
    angular_distance = min(diff, 2 * np.pi - diff)
    if angular_distance > min_angle_separation:
        break
random_angles = np.array([angle1, angle2])

# 計算焦點 F1, F2 的座標
x_foci = CIRCLE_CENTER[0] + CIRCLE_RADIUS * np.cos(random_angles)
y_foci = CIRCLE_CENTER[1] + CIRCLE_RADIUS * np.sin(random_angles)
F1 = np.array([x_foci[0], y_foci[0]])
F2 = np.array([x_foci[1], y_foci[1]])

# 繪製位於圓周上的兩個焦點
ax.plot(x_foci, y_foci, 'ko', markersize=6)

# 畫出連接圓周上兩點的連線 (弦)
ax.plot([F1[0], F2[0]], [F1[1], F2[1]], 'k-', linewidth=1.2)

# ==================== 新增的程式碼在這裡 ====================
# --- 5. 繪製中垂線 (Perpendicular Bisector) ---
# a. 計算弦的中點
midpoint = (F1 + F2) / 2

# b. 計算弦的向量，並找到一個與其垂直的向量
chord_vector = F2 - F1
# 向量 (dx, dy) 的垂直向量是 (-dy, dx)
perp_vector = np.array([-chord_vector[1], chord_vector[0]])

# c. 為了畫出一條足夠長的線，我們先將垂直向量標準化 (長度變為1)
#    然後再乘以一個足夠大的長度，這樣可以避免因 F1,F2 距離太近導致線段過短的問題
norm_perp_vector = perp_vector / np.linalg.norm(perp_vector)
extension_length = CIRCLE_RADIUS * 2 # 讓線的長度是直徑的兩倍，確保能貫穿全圖

# d. 計算線的兩個端點
line_point1 = midpoint + norm_perp_vector * extension_length
line_point2 = midpoint - norm_perp_vector * extension_length

# e. 繪製中垂線 (紅色點虛線)
ax.plot([line_point1[0], line_point2[0]], [line_point1[1], line_point2[1]], 
        'r-.', linewidth=1, label='Perpendicular Bisector') # 'r-.' 表示紅色點虛線
# ==========================================================

# --- 6. 計算並繪製滿足條件的橢圓 ---
# a. 計算焦半距 c
focal_distance_2c = np.linalg.norm(F1 - F2)
c = focal_distance_2c / 2.0

# b. 根據約束 c < a <= R，在這個範圍內隨機選擇 a
lower_bound_a = c * 1.05
upper_bound_a = CIRCLE_RADIUS
if lower_bound_a >= upper_bound_a:
    a = CIRCLE_RADIUS
else:
    a = np.random.uniform(lower_bound_a, upper_bound_a)

# c. 計算短半軸 b
b = np.sqrt(a**2 - c**2)

# d. 計算橢圓的中心和旋轉角度
ellipse_center_new = (F1 + F2) / 2
vec = F2 - F1
rotation_angle = np.arctan2(vec[1], vec[0])

# e. 產生並轉換橢圓的點
t = np.linspace(0, 2 * np.pi, 200)
x_std_ellipse = a * np.cos(t)
y_std_ellipse = b * np.sin(t)
x_rotated = x_std_ellipse * np.cos(rotation_angle) - y_std_ellipse * np.sin(rotation_angle)
y_rotated = x_std_ellipse * np.sin(rotation_angle) + y_std_ellipse * np.cos(rotation_angle)
x_ellipse = x_rotated + ellipse_center_new[0]
y_ellipse = y_rotated + ellipse_center_new[1]

# f. 繪製天藍色實線橢圓
ax.plot(x_ellipse, y_ellipse, color='skyblue', linewidth=2)

# --- 7. 尋找並標示最近點 ---
# a. 計算橢圓上每個點到圓心 (0,0) 的距離的平方
distances_sq = x_ellipse**2 + y_ellipse**2

# b. 找到距離平方最小的那個點的索引
min_distance_index = np.argmin(distances_sq)

# c. 根據索引獲取該點的 x, y 座標
closest_point_x = x_ellipse[min_distance_index]
closest_point_y = y_ellipse[min_distance_index]

# d. 用紅色圓點標示出這個最近的點
ax.plot(closest_point_x, closest_point_y, 'ro', markersize=2)

# --- 8. 清理和顯示 ---
ax.axis('off')
ax.autoscale_view()
ax.set_aspect('equal', adjustable='box')
plt.show()