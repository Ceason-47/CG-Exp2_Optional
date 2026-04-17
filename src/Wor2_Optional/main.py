# main.py
import taichi as ti
import math
from cube_config import *
from transform import *

# 初始化 Taichi
ti.init(arch=ti.cpu)

# 开辟内存：8个顶点，8个投影后的屏幕坐标
vertices = ti.Vector.field(3, dtype=ti.f32, shape=8)
screen_coords = ti.Vector.field(2, dtype=ti.f32, shape=8)

@ti.kernel
def compute_transform(t: ti.f32):
    """
    根据时间参数 t (0到1) 计算当前的 MVP 矩阵并变换顶点
    """
    eye_pos = ti.Vector(EYE_POS)
    view = get_view_matrix(eye_pos)
    proj = get_projection_matrix(FOV, ASPECT, Z_NEAR, Z_FAR)
    
    # 1. 状态定义：起始状态 (R0) 和 目标状态 (R1)
    rot0 = ti.Vector([0.0, 0.0, 0.0])              # 初始无旋转
    rot1 = ti.Vector([180.0, 360.0, 90.0])         # 目标旋转角度
    
    pos0 = ti.Vector([-4.0, -1.0, 0.0])            # 左边起点
    pos1 = ti.Vector([ 4.0, -1.0, 0.0])            # 右边终点
    
    # 2. 插值计算 (Lerp)
    curr_rot = rot0 + (rot1 - rot0) * t
    curr_pos = pos0 + (pos1 - pos0) * t
    
    # 彩蛋：为了模拟参考图片中的“弧线轨迹”，给 Y 轴加上一个基于正弦波的高度
    curr_pos[1] += ti.sin(t * math.pi) * 3.0
    
    # 3. 生成 Model 矩阵 (先旋转，再平移)
    R = get_euler_rotation_matrix(curr_rot[0], curr_rot[1], curr_rot[2])
    T = get_translation_matrix(curr_pos[0], curr_pos[1], curr_pos[2])
    model = T @ R
    
    # 4. MVP 矩阵相乘
    mvp = proj @ view @ model
    
    # 5. 遍历 8 个顶点进行空间变换
    for i in range(8):
        v = vertices[i]
        v4 = ti.Vector([v[0], v[1], v[2], 1.0])
        v_clip = mvp @ v4
        
        # 透视除法
        v_ndc = v_clip / v_clip[3]
        
        # 视口变换映射到 [0, 1] 范围
        screen_coords[i][0] = (v_ndc[0] + 1.0) / 2.0
        screen_coords[i][1] = (v_ndc[1] + 1.0) / 2.0

def main():
    # 初始化顶点数据
    for i in range(8):
        vertices[i] = CUBE_VERTICES[i]
        
    gui = ti.GUI("3D Cube Interpolation", res=RES)
    
    # 动画参数
    time_t = 0.0
    direction = 1.0
    speed = 0.01  # 每次渲染增加的 t 的步长
    
    while gui.running and not gui.get_event(ti.GUI.ESCAPE):
        # 让时间 t 在 0 和 1 之间来回震荡 (Ping-Pong 动画)
        time_t += speed * direction
        if time_t >= 1.0:
            time_t = 1.0
            direction = -1.0
        elif time_t <= 0.0:
            time_t = 0.0
            direction = 1.0
            
        # 并行计算所有顶点的当前位置
        compute_transform(time_t)
        
        # 绘制正方体的 12 条边
        for edge in CUBE_EDGES:
            idx0, idx1 = edge
            p0 = screen_coords[idx0]
            p1 = screen_coords[idx1]
            # 浅蓝色的线框，带一点透视的科幻感
            gui.line(p0, p1, radius=2, color=0x88CCFF)
            
        gui.show()

if __name__ == '__main__':
    main()