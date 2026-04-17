# transform.py
import taichi as ti
import math

@ti.func
def get_euler_rotation_matrix(rx: ti.f32, ry: ti.f32, rz: ti.f32):
    """根据欧拉角(Pitch, Yaw, Roll)生成三维旋转矩阵"""
    rad_x = rx * math.pi / 180.0
    rad_y = ry * math.pi / 180.0
    rad_z = rz * math.pi / 180.0

    cx, sx = ti.cos(rad_x), ti.sin(rad_x)
    cy, sy = ti.cos(rad_y), ti.sin(rad_y)
    cz, sz = ti.cos(rad_z), ti.sin(rad_z)

    # 绕X, Y, Z轴的旋转矩阵
    Rx = ti.Matrix([[1.0, 0.0, 0.0, 0.0],
                    [0.0,  cx, -sx, 0.0],
                    [0.0,  sx,  cx, 0.0],
                    [0.0, 0.0, 0.0, 1.0]])
                    
    Ry = ti.Matrix([[ cy, 0.0,  sy, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [-sy, 0.0,  cy, 0.0],
                    [0.0, 0.0, 0.0, 1.0]])
                    
    Rz = ti.Matrix([[ cz, -sz, 0.0, 0.0],
                    [ sz,  cz, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0]])
                    
    # 返回组合旋转矩阵
    return Rz @ Ry @ Rx

@ti.func
def get_translation_matrix(tx: ti.f32, ty: ti.f32, tz: ti.f32):
    """平移矩阵"""
    return ti.Matrix([
        [1.0, 0.0, 0.0, tx],
        [0.0, 1.0, 0.0, ty],
        [0.0, 0.0, 1.0, tz],
        [0.0, 0.0, 0.0, 1.0]
    ])

@ti.func
def get_view_matrix(eye_pos):
    """视图变换矩阵"""
    return ti.Matrix([
        [1.0, 0.0, 0.0, -eye_pos[0]],
        [0.0, 1.0, 0.0, -eye_pos[1]],
        [0.0, 0.0, 1.0, -eye_pos[2]],
        [0.0, 0.0, 0.0, 1.0]
    ])

@ti.func
def get_projection_matrix(eye_fov: ti.f32, aspect_ratio: ti.f32, zNear: ti.f32, zFar: ti.f32):
    """透视投影矩阵 (原封不动使用基础实验的代码)"""
    n = -zNear
    f = -zFar
    
    fov_rad = eye_fov * math.pi / 180.0
    t = ti.tan(fov_rad / 2.0) * ti.abs(n)
    b = -t
    r = aspect_ratio * t
    l = -r
    
    M_p2o = ti.Matrix([
        [n, 0.0, 0.0, 0.0],
        [0.0, n, 0.0, 0.0],
        [0.0, 0.0, n + f, -n * f],
        [0.0, 0.0, 1.0, 0.0]
    ])
    
    M_ortho_scale = ti.Matrix([
        [2.0 / (r - l), 0.0, 0.0, 0.0],
        [0.0, 2.0 / (t - b), 0.0, 0.0],
        [0.0, 0.0, 2.0 / (n - f), 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ])
    
    M_ortho_trans = ti.Matrix([
        [1.0, 0.0, 0.0, -(r + l) / 2.0],
        [0.0, 1.0, 0.0, -(t + b) / 2.0],
        [0.0, 0.0, 1.0, -(n + f) / 2.0],
        [0.0, 0.0, 0.0, 1.0]
    ])
    
    M_ortho = M_ortho_scale @ M_ortho_trans
    return M_ortho @ M_p2o