import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
import numpy as np
import math

# --- 【核心修改】 ---
# 導入我們自己的裝置生成模組
from Devices import generate_device_locations

# --- 解決中文亂碼問題 ---
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] 
plt.rcParams['axes.unicode_minus'] = False

# --- 全域變數與參數設定 ---
ALL_DRAGGABLE_OBJECTS = []
SELECTED_OBJECT = None
SNAP_RADIUS = 50

# --- 【核心修改】 ---
# 將場景參數統一定義在主程式的開頭
AREA_WIDTH = 2000
AREA_HEIGHT = 2000
COMM_RADIUS = 200
BS_COORDS = (AREA_WIDTH / 2, AREA_HEIGHT / 2)
BS_SIZE = 100

# 動態生成地面裝置座標，並將其設為全域變數
STATIC_NODES_COORDS = generate_device_locations(
    num_nodes=25,
    area_width=AREA_WIDTH,
    area_height=AREA_HEIGHT,
    comm_radius=COMM_RADIUS,
    min_node_distance=100.0,
    bs_coords=BS_COORDS,
    bs_size=BS_SIZE
)

# (從這裡開始，後面的類別和函數定義與上一版完全相同)
def set_selected(obj):
    global SELECTED_OBJECT
    if SELECTED_OBJECT: SELECTED_OBJECT.set_selected(False)
    SELECTED_OBJECT = obj
    if SELECTED_OBJECT: SELECTED_OBJECT.set_selected(True)
    if obj and hasattr(obj, 'artist'): obj.artist.figure.canvas.draw_idle()
def on_key_press(event):
    if event.key == 'delete' and SELECTED_OBJECT:
        SELECTED_OBJECT.delete(); set_selected(None)
def get_all_snap_targets(exclude_object=None):
    targets = [BS_COORDS] + STATIC_NODES_COORDS
    for obj in ALL_DRAGGABLE_OBJECTS:
        if obj is exclude_object: continue
        if isinstance(obj, DraggablePoint):
            x_data, y_data = obj.artist.get_data(); targets.append((x_data[0], y_data[0]))
    return targets
def find_snap_target(x, y, targets):
    min_dist = float('inf'); snap_pos = None
    for tx, ty in targets:
        dist = math.sqrt((x - tx)**2 + (y - ty)**2)
        if dist < SNAP_RADIUS and dist < min_dist: min_dist = dist; snap_pos = (tx, ty)
    return snap_pos
class DraggablePoint:
    def __init__(self, ax, x, y, size=4, color='yellow'):
        self.ax = ax; self.size = size; self.color = color
        self.artist, = ax.plot([x], [y], 'o', color=self.color, markersize=self.size, markeredgecolor='black', markeredgewidth=0.5, zorder=10, picker=True, pickradius=10)
        self.is_dragging = False; self.connect()
    def set_selected(self, is_selected):
        if is_selected: self.artist.set_markersize(self.size * 2); self.artist.set_markeredgewidth(2)
        else: self.artist.set_markersize(self.size); self.artist.set_markeredgewidth(0.5)
    def connect(self): self.cids = [self.artist.figure.canvas.mpl_connect(event, getattr(self, handler)) for event, handler in [('button_press_event', 'on_press'), ('motion_notify_event', 'on_motion'), ('button_release_event', 'on_release')]]
    def on_press(self, event):
        if event.inaxes != self.artist.axes: set_selected(None); return
        contains, _ = self.artist.contains(event)
        if not contains:
            if self is SELECTED_OBJECT: set_selected(None); return
        set_selected(self); self.is_dragging = True
    def on_motion(self, event):
        if not self.is_dragging or event.inaxes is None: return
        x, y = event.xdata, event.ydata
        targets = get_all_snap_targets(exclude_object=self); snap_pos = find_snap_target(x, y, targets)
        if snap_pos: x, y = snap_pos
        self.artist.set_data([x], [y]); self.artist.figure.canvas.draw_idle()
    def on_release(self, event):
        if not self.is_dragging: return
        self.is_dragging = False; self.on_motion(event)
    def delete(self):
        self.artist.remove()
        for cid in self.cids: self.artist.figure.canvas.mpl_disconnect(cid)
        if self in ALL_DRAGGABLE_OBJECTS: ALL_DRAGGABLE_OBJECTS.remove(self)
        self.artist.figure.canvas.draw_idle()
class DraggableLine:
    def __init__(self, ax, p1, p2, color='red', linewidth=2):
        self.ax = ax; self.p1 = list(p1); self.p2 = list(p2); self.color = color; self.linewidth = linewidth
        self.line = Line2D([self.p1[0], self.p2[0]], [self.p1[1], self.p2[1]], color=self.color, linewidth=self.linewidth, zorder=9, picker=True, pickradius=5)
        self.handle1, = ax.plot(self.p1[0], self.p1[1], 's', markersize=0, zorder=10, picker=True, pickradius=10)
        self.handle2, = ax.plot(self.p2[0], self.p2[1], 's', markersize=0, zorder=10, picker=True, pickradius=10)
        ax.add_line(self.line); self.drag_target = None; self.press_data = None; self.connect()
    def set_selected(self, is_selected):
        if is_selected: self.line.set_linewidth(self.linewidth * 2); self.line.set_path_effects([plt.matplotlib.patheffects.withStroke(linewidth=self.linewidth*3, foreground='cyan')])
        else: self.line.set_linewidth(self.linewidth); self.line.set_path_effects(None)
    def connect(self): self.cids = [self.line.figure.canvas.mpl_connect(event, getattr(self, handler)) for event, handler in [('button_press_event', 'on_press'), ('motion_notify_event', 'on_motion'), ('button_release_event', 'on_release')]]
    def on_press(self, event):
        if event.inaxes != self.ax: set_selected(None); return
        contains_h1, _ = self.handle1.contains(event); contains_h2, _ = self.handle2.contains(event); contains_line, _ = self.line.contains(event)
        if not (contains_h1 or contains_h2 or contains_line):
            if self is SELECTED_OBJECT: set_selected(None); return
        set_selected(self)
        if contains_h1: self.drag_target = 'handle1'
        elif contains_h2: self.drag_target = 'handle2'
        elif contains_line: self.drag_target = 'line'
        self.press_data = (event.xdata, event.ydata, self.p1[:], self.p2[:])
    def on_motion(self, event):
        if self.drag_target is None or event.inaxes is None: return
        x, y, p1_orig, p2_orig = self.press_data; dx = event.xdata - x; dy = event.ydata - y
        targets = get_all_snap_targets(exclude_object=self)
        if self.drag_target == 'handle1':
            snap_pos = find_snap_target(event.xdata, event.ydata, targets)
            self.p1 = snap_pos if snap_pos else [event.xdata, event.ydata]
        elif self.drag_target == 'handle2':
            snap_pos = find_snap_target(event.xdata, event.ydata, targets)
            self.p2 = snap_pos if snap_pos else [event.xdata, event.ydata]
        elif self.drag_target == 'line': self.p1 = [p1_orig[0] + dx, p1_orig[1] + dy]; self.p2 = [p2_orig[0] + dx, p2_orig[1] + dy]
        self.update_artists()
    def on_release(self, event):
        if self.drag_target is None: return
        self.on_motion(event); self.drag_target = None; self.press_data = None
    def delete(self):
        self.line.remove(); self.handle1.remove(); self.handle2.remove()
        for cid in self.cids: self.line.figure.canvas.mpl_disconnect(cid)
        if self in ALL_DRAGGABLE_OBJECTS: ALL_DRAGGABLE_OBJECTS.remove(self)
        self.line.figure.canvas.draw_idle()
    def update_artists(self):
        if self.line.figure is None: return
        self.line.set_data([self.p1[0], self.p2[0]], [self.p1[1], self.p2[1]])
        self.handle1.set_data([self.p1[0]], [self.p1[1]]); self.handle2.set_data([self.p2[0]], [self.p2[1]])
        self.line.figure.canvas.draw_idle()
class ObjectFactory:
    def __init__(self, fig, main_ax, toolbox_ax): self.fig=fig; self.main_ax=main_ax; self.toolbox_ax=toolbox_ax; self.connect()
    def connect(self): self.cid_press = self.fig.canvas.mpl_connect('button_press_event', self.on_press)
    def on_press(self, event):
        if event.inaxes != self.toolbox_ax: return
        for template in TOOLBOX_TEMPLATES:
            contains, _ = template['artist'].contains(event)
            if contains:
                obj_type, color = template['type'], template['color']
                if obj_type == 'point': obj = DraggablePoint(self.main_ax, 100, 100, color=color)
                elif obj_type == 'line': obj = DraggableLine(self.main_ax, (80, 100), (120, 100), color=color)
                set_selected(obj); obj.is_dragging = True if obj_type == 'point' else False
                if obj_type == 'line': obj.drag_target = 'line'; obj.press_data = (100, 100, obj.p1[:], obj.p2[:])
                ALL_DRAGGABLE_OBJECTS.append(obj); self.fig.canvas.draw_idle(); break
def draw_static_scenario_map(ax):
    # 【核心修改】這個函數現在使用全域變數來繪製場景
    if not STATIC_NODES_COORDS: return
    node_x_coords, node_y_coords = zip(*STATIC_NODES_COORDS)
    bs_bottom_left = (BS_COORDS[0] - BS_SIZE / 2, BS_COORDS[1] - BS_SIZE / 2)
    base_station_patch = patches.Rectangle(bs_bottom_left, BS_SIZE, BS_SIZE, facecolor='green', edgecolor='black', linewidth=1.5, label='基地台 (BS)', zorder=2); ax.add_patch(base_station_patch)
    ax.plot(node_x_coords, node_y_coords, 'ko', markersize=5, label='地面裝置', zorder=3)
    for x, y in STATIC_NODES_COORDS:
        communication_range = patches.Circle((x, y), COMM_RADIUS, linestyle='--', edgecolor='gray', facecolor='lightgray', alpha=0.4, zorder=1)
        ax.add_patch(communication_range)
    ax.set_xlim(0, AREA_WIDTH); ax.set_ylim(0, AREA_HEIGHT); ax.set_aspect('equal', adjustable='box')
    ax.set_title('互動式路徑規劃 (點擊選中, Delete鍵刪除)'); ax.set_xlabel('X 座標 (公尺)'); ax.set_ylabel('Y 座標 (公尺)')
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.plot([], [], '--', color='gray', label=f'通訊範圍 (半徑={COMM_RADIUS}m)')
def draw_toolbox(ax):
    ax.set_facecolor('whitesmoke'); ax.spines[:].set_visible(False); ax.set_xticks([]); ax.set_yticks([]); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.text(0.5, 0.9, '工具箱', ha='center', fontsize=12)
    global TOOLBOX_TEMPLATES; TOOLBOX_TEMPLATES = []
    point_artist, = ax.plot(0.3, 0.8, 'o', color='yellow', markeredgecolor='black', markeredgewidth=0.5, markersize=10, picker=True, pickradius=15)
    ax.text(0.5, 0.8, '拖曳點', va='center', ha='left'); TOOLBOX_TEMPLATES.append({'artist': point_artist, 'type': 'point', 'color': 'yellow'})
    line_artist_red = Line2D([0.2, 0.4], [0.65, 0.65], color='red', linewidth=3, picker=True, pickradius=15); ax.add_line(line_artist_red)
    ax.text(0.5, 0.65, '紅色路徑', va='center', ha='left'); TOOLBOX_TEMPLATES.append({'artist': line_artist_red, 'type': 'line', 'color': 'red'})
    line_artist_blue = Line2D([0.2, 0.4], [0.45, 0.45], color='blue', linewidth=3, picker=True, pickradius=15); ax.add_line(line_artist_blue)
    ax.text(0.5, 0.45, '藍色路徑', va='center', ha='left'); TOOLBOX_TEMPLATES.append({'artist': line_artist_blue, 'type': 'line', 'color': 'blue'})
    line_artist_black = Line2D([0.2, 0.4], [0.25, 0.25], color='black', linewidth=3, picker=True, pickradius=15); ax.add_line(line_artist_black)
    ax.text(0.5, 0.25, '黑色路徑', va='center', ha='left'); TOOLBOX_TEMPLATES.append({'artist': line_artist_black, 'type': 'line', 'color': 'black'})

if __name__ == '__main__':
    fig = plt.figure(figsize=(12, 10))
    ax_main = fig.add_axes([0.08, 0.1, 0.7, 0.8])
    ax_toolbox = fig.add_axes([0.8, 0.6, 0.18, 0.3])
    
    fig.canvas.mpl_connect('key_press_event', on_key_press)
    
    draw_static_scenario_map(ax_main)
    draw_toolbox(ax_toolbox)
    factory = ObjectFactory(fig, ax_main, ax_toolbox)
    
    # --- 【核心修改】 ---
    # 將 fig.legend(...) 改為 ax_main.legend(...)
    # 這樣圖例只會被繪製在主地圖的座標軸內，不會影響到工具箱的佈局
    handles, labels = ax_main.get_legend_handles_labels()
    ax_main.legend(handles, labels, loc='upper left') # 使用 ax_main.legend
    
    plt.show()