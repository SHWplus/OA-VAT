"""无人机控制相关函数"""
import sys
import numpy as np


def compute_drone_action_while_tracking(mean_point, current_area, cfg, vehicle, pid_controllers,  
                                       is_predict=False, frame_idx=None):
    """计算并发送无人机控制命令（跟踪阶段）
    
    Args:
        mean_point: 目标中心点 (cy, cx)
        current_area: 当前目标面积
        cfg: 配置字典
        vehicle: 无人机控制器实例
        pid_controllers: PID控制器字典 {'yaw': ..., 'vz': ..., 'vy': ...}
        is_predict: 是否为预测模式
        frame_idx: 帧索引（可选，用于调试）
    """
    if vehicle is None:
        return
        
    if mean_point is None:
        print("目标丢失，暂停控制指令")
        vehicle.send_velocity(0, 0, 0, 0)
        return
        
    # 检查电量
    try:
        if vehicle.tello.get_battery() < int(cfg.get('ctrl_low_battery', 15)):
            print("电量低于15%，紧急降落！")
            vehicle.land()
            sys.exit(1)
    except Exception:
        pass

    frame_width = cfg['desired_width']
    frame_height = cfg['desired_height']
    max_speed = int(cfg.get('ctrl_max_speed', 40))
    
    # 以帧面积比例定义期望面积 + 相对容差
    frame_area = int(frame_width * frame_height)
    target_area = int(float(cfg.get('target_area', cfg.get('target_area_ratio', 0.12))) * frame_area)
    tol = int(float(cfg.get('area_tolerance_ratio', 0.20)) * max(1, target_area))

    if is_predict:
        print("[预测模式] 使用卡尔曼滤波预测点控制无人机")

    # 计算误差
    error_x = mean_point[1] - (frame_width // 2)  # 水平误差（偏航）
    error_y = (frame_height // 2) - mean_point[0]   # 垂直误差
    
    # PID 控制
    pid_yaw = pid_controllers['yaw']
    pid_vz = pid_controllers['vz']
    pid_vy = pid_controllers['vy']
    
    yaw_raw = pid_yaw.update(error_x, dt=float(cfg.get('ctrl_dt', 0.1)))
    vz_raw = 0.0  # 禁用垂直控制
    
    # 连续面积误差控制前后速度，落在容差内则置零
    area_error = target_area - int(current_area)
    vy_raw = 0.0
    if abs(area_error) > tol:
        vy_raw = pid_vy.update(area_error, dt=float(cfg.get('ctrl_dt', 0.1)))
        # 反死区：保持小幅前进/后退避免停滞
        vy_min = float(cfg.get('ctrl_vy_min_output', 0.8))
        if vy_raw > 0:
            vy_raw = max(vy_raw, vy_min)
        elif vy_raw < 0:
            vy_raw = min(vy_raw, -vy_min)

    # 饱和限制
    yaw = float(np.clip(yaw_raw, -max_speed, max_speed))
    vz = float(np.clip(vz_raw, -max_speed, max_speed))
    vy = float(np.clip(vy_raw, -max_speed, max_speed))

    # 打印控制参数（仅首次）
    if cfg.get('debug_control', False) and not cfg.get('_ctrl_cfg_printed', False):
        try:
            print(f"[CTRL-CFG] dt={cfg.get('ctrl_dt', 0.1)} max_speed={max_speed} "
                  f"target_area_ratio={cfg.get('target_area_ratio', 0.12)} "
                  f"tol_ratio={cfg.get('area_tolerance_ratio', 0.20)}")
            print(f"[CTRL-CFG] yaw(Kp,Ki,Kd)=({pid_yaw.Kp},{pid_yaw.Ki},{pid_yaw.Kd}) "
                  f"vz=({pid_vz.Kp},{pid_vz.Ki},{pid_vz.Kd}) "
                  f"vy=({pid_vy.Kp},{pid_vy.Ki},{pid_vy.Kd})")
            cfg['_ctrl_cfg_printed'] = True
        except Exception:
            pass

    # 打印当前帧控制信息
    if cfg.get('debug_control', False):
        try:
            tag = f"f={frame_idx}" if frame_idx is not None else ""
            print(f"[CTRL] {tag} ex={error_x} ey={error_y} area={int(current_area)} "
                  f"target={target_area} tol=±{tol} | yaw={yaw:.1f}({yaw_raw:.1f}) "
                  f"vz={vz:.1f}({vz_raw:.1f}) vy={vy:.1f}({vy_raw:.1f})")
        except Exception:
            pass

    # 发送控制命令
    try:
        # 记录本帧控制量，便于详细日志
        cfg['_last_cmd'] = {
            'vy': float(vy),
            'yaw': float(yaw),
            'sat_vy': abs(float(vy)) >= float(cfg.get('ctrl_max_speed', 40)) - 1e-6,
            'sat_yaw': abs(float(yaw)) >= float(cfg.get('ctrl_max_speed', 40)) - 1e-6,
            'back': float(vy) < 0
        }
        vehicle.send_velocity(
            vx=0,
            vy=vy,
            vz=vz,
            yaw_rate=yaw
        )
    except Exception as e:
        print(f"发送无人机速度指令失败: {e}")




