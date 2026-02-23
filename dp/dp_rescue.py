"""Diffusion Policy 补救逻辑"""
import time
import numpy as np
import torch


def dp_reposition_and_try_reacquire(video, vehicle, detector, feature_extractor, ref_store, cfg,
                                   pred_bbox_xywh=None, dp_cache=None, read_and_log_fn=None,
                                   find_instance_by_features_fn=None, save_follow_stream_frame_fn=None,
                                   load_dp_policy_fn=None, build_dp_obs_fn=None, dp_action_to_vel_fn=None):
    """DP补救：使用Diffusion Policy规划补救以打破遮挡
    
    RHC（论文 2.3）：每次规划得到一段动作序列（Tp），仅执行其中前 N 步（Ta=--dp_action_exec_steps），
    丢弃剩余，随后基于新观测重新规划。完全按步数滚动，不使用时间窗口。
    
    Args:
        video: 视频源
        vehicle: 无人机控制器
        detector: YOLO检测器
        feature_extractor: 特征提取器
        ref_store: 参考特征存储
        cfg: 配置字典
        pred_bbox_xywh: 预测框 (x, y, w, h)
        dp_cache: DP缓存字典 {'policy': ..., 'cfg': ..., 'device': ...}
        read_and_log_fn: 读帧函数
        find_instance_by_features_fn: 特征匹配函数
        save_follow_stream_frame_fn: 保存跟踪流帧函数
        load_dp_policy_fn: DP策略加载函数
        build_dp_obs_fn: DP观测构建函数
        dp_action_to_vel_fn: DP动作转速度函数
        
    Returns:
        tuple: (success, detected_bbox)
            - success: 是否成功找回目标
            - detected_bbox: 检测到的边界框（如果找回）
    """
    dt = float(cfg['dp_dt']) if cfg.get('dp_dt') is not None else float(cfg.get('ctrl_dt', 0.1))

    policy, dp_cfg, device = dp_cache['policy'], dp_cache['cfg'], dp_cache['device']
    try:
        print(f"[DP] enable={bool(cfg.get('dp_enable', False))}, ckpt={cfg.get('dp_checkpoint','')}, cached_policy={policy is not None}")
    except Exception:
        pass
        
    if policy is None and cfg.get('dp_enable', False) and cfg.get('dp_checkpoint'):
        pol, pcfg, dev = load_dp_policy_fn(cfg['dp_checkpoint'], cfg.get('dp_repo_root', ''))
        if pol is not None:
            dp_cache.update({'policy': pol, 'cfg': pcfg, 'device': dev})
            policy, dp_cfg, device = pol, pcfg, dev

    # 打印步数滚动配置
    try:
        print(f"[DP] RHC mode: dp_dt={dt}, exec_steps={cfg.get('dp_action_exec_steps', None)}, max_plans={cfg.get('dp_max_plans', 0)}")
    except Exception:
        pass

    action_seq = None
    ptr = 0
    plan_cnt = 0  # 本次DP补救内的规划次数
    seq_step_count = 0  # 当前action_seq已执行步数
    
    try:
        print("[DP] pre-zero RC to clear previous command")
        vehicle.send_velocity(0, 0, 0, 0)
    except Exception:
        pass

    # 纯步数滚动：不看时间，直到命中 max_plans 或找到目标
    while True:
        try:
            print(f"[DP] loop enter: plan_cnt={plan_cnt}, action_seq={'set' if action_seq is not None else 'None'}, ptr={ptr}")
        except Exception:
            pass
            
        if policy is not None and (action_seq is None or ptr >= len(action_seq)):
            ok, fr = read_and_log_fn(video, cfg)
            if not ok:
                try:
                    print("[DP] video.read() returned not ok; abort DP loop")
                except Exception:
                    pass
                break
                
            if pred_bbox_xywh is None:
                H, W = fr.shape[:2]
                cw, ch = int(W * 0.2), int(H * 0.2)
                pred = (W//2 - cw//2, H//2 - ch//2, cw, ch)
            else:
                pred = tuple(int(v) for v in pred_bbox_xywh)
                
            # FollowStream：保存 DP 规划输入帧（带当前预测框）
            try:
                save_follow_stream_frame_fn(cfg, fr, stage="DP", predict_bbox=pred)
            except Exception:
                pass
                
            obs = build_dp_obs_fn(fr, pred, dp_cfg, device)
            try:
                print(f"[DP] obs shapes: image={tuple(obs['image'].shape)}, bbox={tuple(obs['bbox'].shape)}")
            except Exception:
                pass
                
            with torch.no_grad():
                try:
                    out = policy.predict_action(obs)
                    try:
                        act = out.get('action', None)
                        print(f"[DP] predict_action OK. action type={type(act)}, shape={tuple(act.shape) if hasattr(act,'shape') else None}")
                    except Exception:
                        pass
                        
                    a = out['action'][0].detach().cpu().numpy()  # (Ta,Da)
                    
                    # 1) 位置序列：以首点为原点做平移，得到相对原点的位置 pos_seq
                    pos_seq = a
                    if pos_seq.ndim == 2 and pos_seq.shape[0] >= 1:
                        if pos_seq.shape[1] >= 3:
                            p0 = pos_seq[0:1, :3]
                            pos_seq = pos_seq.copy()
                            pos_seq[:, :3] = pos_seq[:, :3] - p0
                            # 2) 增量序列：相邻差分，第一步为0
                            delta_seq = pos_seq.copy()
                            delta_seq[1:, :3] = pos_seq[1:, :3] - pos_seq[:-1, :3]
                            delta_seq[0, :3] = 0.0
                        else:
                            p0 = pos_seq[0:1, :2]
                            pos_seq = pos_seq.copy()
                            pos_seq[:, :2] = pos_seq[:, :2] - p0
                            delta_seq = pos_seq.copy()
                            delta_seq[1:, :2] = pos_seq[1:, :2] - pos_seq[:-1, :2]
                            delta_seq[0, :2] = 0.0
                    else:
                        delta_seq = pos_seq
                        
                    # 用增量序列来执行；位置序列仅用于日志
                    action_pos_seq = pos_seq
                    action_seq = delta_seq
                    ptr = 0
                    seq_step_count = 0
                    
                    # 打印位置与增量的前若干步
                    try:
                        print(f"[DP] New action_seq predicted: shape={action_seq.shape}")
                        K = min(5, action_seq.shape[0])
                        head_pos = action_pos_seq[:K, :]
                        head_delta = action_seq[:K, :]
                        print("[DP] pos_seq head:\n" + 
                              np.array2string(head_pos, formatter={'float_kind': lambda x: f'{x:.3f}'}))
                        print("[DP] delta_seq head:\n" + 
                              np.array2string(head_delta, formatter={'float_kind': lambda x: f'{x:.3f}'}))
                        cmds = []
                        for i in range(K):
                            vy_i, vz_i, yaw_i = dp_action_to_vel_fn(action_seq[i], cfg)
                            cmds.append([vy_i, vz_i, yaw_i])
                        print("[DP] planned first {} RC (vy,vz,yaw):\n".format(K) + 
                              np.array2string(np.array(cmds), formatter={'float_kind': lambda x: f'{x:.2f}'}))
                    except Exception:
                        pass
                        
                    plan_cnt += 1
                    # 若设置了单次最多规划次数，则到上限后不再触发新规划
                    max_plans = int(cfg.get('dp_max_plans', 0) or 0)
                    if max_plans > 0 and plan_cnt >= max_plans:
                        try:
                            print(f"[DP] Hit dp_max_plans={max_plans}, plan_cnt={plan_cnt}. Stop replanning after current sequence.")
                        except Exception:
                            pass
                        # 不立刻 break，允许当前 sequence 按 exec_steps 执行完
                except Exception as e:
                    print(f"[DP] 推理失败，跳过本次DP: {e}")
                    action_seq = None

        if policy is not None and action_seq is not None and ptr < len(action_seq):
            try:
                raw_delta = action_seq[ptr]
                print("[DP] step raw delta:", 
                      np.array2string(raw_delta, formatter={'float_kind': lambda x: f'{x:.3f}'}))
                try:
                    if 'action_pos_seq' in locals() and ptr < len(action_pos_seq):
                        raw_pos = action_pos_seq[ptr]
                        print("[DP] step raw pos  :", 
                              np.array2string(raw_pos, formatter={'float_kind': lambda x: f'{x:.3f}'}))
                except Exception:
                    pass
            except Exception:
                pass
            vy, vz, yaw = dp_action_to_vel_fn(action_seq[ptr], cfg)
            ptr += 1
        else:
            try:
                reason = []
                if policy is None: 
                    reason.append("policy=None")
                if action_seq is None: 
                    reason.append("action_seq=None")
                elif ptr >= len(action_seq): 
                    reason.append(f"ptr({ptr})>=len({len(action_seq)})")
                print(f"[DP][IDLE-REASON] {'; '.join(reason)}")
            except Exception:
                pass
            vy, vz, yaw = 0.0, 0.0, 0.0

        try:
            vehicle.send_velocity(vx=0.0, vy=vy, vz=vz, yaw_rate=yaw)
        except Exception:
            pass

        # 打印本步执行的控制指令
        try:
            if policy is not None and action_seq is not None and ptr > 0 and (ptr - 1) < len(action_seq):
                i = ptr - 1
                print(f"[DP] exec step {i+1}/{len(action_seq)}: vy={vy:.2f}, vz={vz:.2f}, yaw={yaw:.2f}")
            else:
                print(f"[DP] exec idle: vy={vy:.2f}, vz={vz:.2f}, yaw={yaw:.2f}")
        except Exception:
            pass

        # 用"读帧排空+按时间门限保存"替代硬等待，避免 DP 阶段跳帧
        t_end = time.time() + dt
        # 频率配置
        read_hz = float(cfg.get('dp_read_hz', 0.0) or 0.0)
        if read_hz <= 0.0:
            read_hz = float(cfg.get('fps', 0.0) or 30.0)
        save_hz = float(cfg.get('dp_save_hz', 0.0) or 0.0)
        if save_hz <= 0.0:
            save_hz = read_hz
        read_dt = 1.0 / max(1.0, read_hz)
        save_dt = 1.0 / max(1.0, save_hz)
        last_read_ts = time.time() - read_dt
        last_save_ts = time.time() - save_dt
        
        while time.time() < t_end:
            now_d = time.time()
            if now_d - last_read_ts >= read_dt:
                ok_drain, fr_drain = video.read()
                last_read_ts = now_d
                if ok_drain and (now_d - last_save_ts >= save_dt):
                    try:
                        save_follow_stream_frame_fn(cfg, fr_drain, stage="DP")
                    except Exception:
                        pass
                    last_save_ts = now_d
            else:
                time.sleep(min(0.002, max(0.0, t_end - now_d)))

        ok2, fr2 = read_and_log_fn(video, cfg, stage="DP")
        if not ok2:
            continue
            
        ref_list = ref_store.get_features() if (ref_store is not None and ref_store.is_valid()) else None
        if not ref_list:
            continue
            
        detected, det_bbox = find_instance_by_features_fn(fr2, detector, feature_extractor, ref_list, cfg, 
                                                          stage="dp_reposition")
        if detected and det_bbox is not None:
            try:
                vehicle.send_velocity(0, 0, 0, 0)
            except Exception:
                pass
            # FollowStream：保存 DP 找回成功帧（画回捕获框）
            try:
                save_follow_stream_frame_fn(cfg, fr2, stage="DP_Found", track_bbox=det_bbox)
            except Exception:
                pass
            return True, det_bbox

        # 仅执行前N步(Ta)后强制重规划（RHC）
        exec_limit = int(cfg.get('dp_action_exec_steps', 0) or 0)
        if policy is not None and action_seq is not None and exec_limit > 0:
            seq_step_count += 1
            if seq_step_count >= exec_limit:
                try:
                    print(f"[DP] reached exec_limit={exec_limit}, forcing replan")
                except Exception:
                    pass
                ptr = len(action_seq)
                # 若已达到最大规划次数，则退出整体循环
                max_plans = int(cfg.get('dp_max_plans', 0) or 0)
                if max_plans > 0 and plan_cnt >= max_plans:
                    break
                continue

    try:
        vehicle.send_velocity(0, 0, 0, 0)
    except Exception:
        pass
    return False, None






