#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Table1 评测脚本（适配重构后的模块入口 main.py）

用法与原先相同：把原本传给 follow2.py 的参数（包括 --eval_table1、--episodes 等）
原样传入本脚本即可。

示例：
  python eval_table1_modular.py \
    --path_to_video "unrealcv:UnrealTrackMulti-UrbanCityNav-ContinuousColor-v0" \
    --episodes 20 --desired_width 160 --desired_height 120 --text_query "person" \
    --online_enhance --reference_feature_path queries/people/feature_set_dinov3.pt \
    --yolo_confidence_thresh 0.1 --reacq_match_thresh 0.5 --match_similarity_thresh 0.4 \
    --ortrack_lost_thresh 0.3 --distractors 2 --target_area_ratio 0.1 --area_tolerance_ratio 0.1 \
    --ctrl_dt 0.05 --ctrl_max_speed 99 --ctrl_yaw_kp 0.1 --ctrl_yaw_kd 0.12 \
    --ctrl_vy_kp 0.1 --ctrl_vy_kd 0.0 --kf_max_predict_frames 3 --eval_log_detail \
    --yolo_model_path "ORTrack/weights/yoloe-11l-seg.pt" --debug_control --kf_mode none \
    --save_images_to ./test_outputs
"""

import os
import sys
import time
import json
import signal
import threading

GYM_UNREALCV_PATH = os.path.expanduser("/home/ws0/gh/follow/gym-unrealcv/gym-unrealcv")
if os.path.isdir(GYM_UNREALCV_PATH) and GYM_UNREALCV_PATH not in sys.path:
    sys.path.insert(0, GYM_UNREALCV_PATH)


def ensure_eval_flag_in_argv():
    if '--eval_table1' not in sys.argv:
        sys.argv.insert(1, '--eval_table1')


def main():
    # 安装 SIGINT/SIGTERM 处理，确保 Ctrl-C 可立即生效
    _video_holder = {'video': None}
    _exit_event = threading.Event()

    def _cleanup_and_exit(exit_code=130):
        _exit_event.set()
        v = _video_holder.get('video', None)

        # 主动杀死 Unreal 引擎子进程
        unreal_pid = None
        try:
            base = getattr(getattr(v, 'env', None), 'unwrapped', getattr(v, 'env', None)) if v is not None else None
            # 尝试获取 Unreal 进程的 PID
            if hasattr(base, 'unreal') and hasattr(base.unreal, 'env') and hasattr(base.unreal.env, 'pid'):
                unreal_pid = int(base.unreal.env.pid)
        except Exception:
            pass

        # 先尝试优雅关闭
        try:
            base = getattr(getattr(v, 'env', None), 'unwrapped', getattr(v, 'env', None)) if v is not None else None
            if hasattr(base, 'close'):
                try:
                    base.close()
                except Exception:
                    pass
            if v is not None and hasattr(v, 'release'):
                try:
                    v.release()
                except Exception:
                    pass
        except Exception:
            pass

        # 强制杀死 Unreal 进程（包括其所有子进程）
        if unreal_pid is not None:
            try:
                import subprocess
                # 杀死进程组（包括所有子进程）
                try:
                    os.killpg(os.getpgid(unreal_pid), signal.SIGKILL)
                except Exception:
                    pass
                # 兜底：直接杀死主进程
                try:
                    os.kill(unreal_pid, signal.SIGKILL)
                except Exception:
                    pass
            except Exception:
                pass

        # 兜底：通过进程名查找并杀死所有 urbancity/Unreal 进程
        try:
            import subprocess
            # 查找包含 "urbancity" 或 "Binaries/Linux" 的进程
            result = subprocess.run(['pgrep', '-f', 'urbancity'], capture_output=True, text=True, timeout=1)
            pids = result.stdout.strip().split('\n')
            for pid in pids:
                if pid and pid.isdigit():
                    try:
                        os.kill(int(pid), signal.SIGKILL)
                    except Exception:
                        pass
        except Exception:
            pass

        # 尝试关闭所有 OpenCV 窗口（若有）
        try:
            import cv2  # 延迟导入，避免无依赖时异常
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
        except Exception:
            pass

        # 直接退出，避免子库阻塞
        os._exit(int(exit_code))

    def _sig_handler(signum, frame):
        _exit_event.set()
        try:
            print(f"[Eval-Table1] Caught signal {signum}, force killing process...", flush=True)
        except Exception:
            pass
        # 直接发送 SIGKILL，无法被捕获或阻塞
        try:
            os.kill(os.getpid(), signal.SIGKILL)
        except Exception:
            pass
        # 若 SIGKILL 失败（罕见），尝试 os._exit
        try:
            os._exit(130 if signum == signal.SIGINT else 143)
        except Exception:
            pass

    try:
        signal.signal(signal.SIGINT, _sig_handler)
        signal.signal(signal.SIGTERM, _sig_handler)
    except Exception:
        pass

    ensure_eval_flag_in_argv()
    # 导入 main，此时其内部会按本脚本的 sys.argv 解析参数
    import main as f

    # 初始化系统（与 follow_modular 相同）
    device, cfg, tracker, detector, video, vehicle, feature_extractor = f.init_system()
    _video_holder['video'] = video

    # 仅处理 UnrealCV 源
    if not isinstance(video, f.UnrealCVStream):
        # 非 UnrealCV 源：单次 start_mission，也可被 Ctrl-C 中断
        def _abort_cb(*_args, **_kwargs):
            return not _exit_event.is_set()
        try:
            f.start_mission(device, cfg, tracker, detector, video, vehicle, feature_extractor, step_cb=_abort_cb)
        except KeyboardInterrupt:
            _cleanup_and_exit(130)
        _cleanup_and_exit(0)
        return

    runs = int(cfg.get('episodes', 100))

    def run_eval_loop(tag):
        # 日志文件：优先写入 --save_images_to 指定目录，否则写入当前目录
        try:
            log_dir = cfg['save_images_to'] if cfg.get('save_images_to') else '.'
        except Exception:
            log_dir = '.'
        try:
            os.makedirs(log_dir, exist_ok=True)
        except Exception:
            pass
        log_path = os.path.join(log_dir, 'eval_table1_metrics.txt')
        try:
            if not os.path.exists(log_path):
                with open(log_path, 'a', encoding='utf-8') as f_out:
                    f_out.write('# tag, episode, AR, EL, SR, status, timestamp\n')
        except Exception:
            pass

        local_sum_ar, local_sum_el, local_sum_sr = 0.0, 0, 0.0

        for ep in range(runs):
            if _exit_event.is_set():
                break
            # 每个 episode：先固定人数，再固定种子，最后 reset，确保随机序列与人口一致
            try:
                base = getattr(video.env, 'unwrapped', video.env)
                # 固定人数（跟踪者+目标+干扰者）
                try:
                    if cfg.get('distractors') is not None and hasattr(base, 'player_num'):
                        base.player_num = int(2 + int(cfg.get('distractors')))
                except Exception:
                    pass
                # 固定随机种子（每个 episode 使用不同但可复现的种子）
                try:
                    if cfg.get('unreal_seed') is not None and hasattr(base, 'seed'):
                        try:
                            base_seed = int(cfg['unreal_seed'])
                        except Exception:
                            base_seed = 0
                        episode_seed = int(base_seed + ep)
                        base.seed(episode_seed)
                except Exception:
                    pass
                # 重置以生效
                try:
                    video.env.reset()
                except Exception:
                    pass
                # 将控制周期与引擎 interval 对齐，避免节拍打架
                try:
                    if hasattr(base, 'interval'):
                        cfg['ctrl_dt'] = float(base.interval)
                except Exception:
                    pass
                # 给初始化旋转一点时间完成（rotate2exp在UE侧需要几个tick）
                try:
                    import time as _time
                    _time.sleep(0.2)
                except Exception:
                    pass
                # 动作日志已禁用（节省磁盘空间）
                try:
                    setattr(base, 'action_log_enabled', False)
                    setattr(base, 'action_log_path', None)
                except Exception:
                    pass
                # 固定可控体数量与外形（不改变人口/种子）
                try:
                    if hasattr(base, 'controable_agent'):
                        base.controable_agent = 1
                    if hasattr(base, 'unrealcv') and hasattr(base, 'player_list'):
                        ucv = base.unrealcv
                        players = list(getattr(base, 'player_list', []))
                        if len(players) >= 2:
                            try:
                                ucv.set_appearance(players[1], id=int(cfg.get('target_appearance_id', 2)), spline=True)
                            except Exception:
                                try:
                                    ucv.set_appearance(players[1], id=int(cfg.get('target_appearance_id', 2)), spline=False)
                                except Exception:
                                    pass
                        for name in players[2:]:
                            try:
                                ucv.set_appearance(name, id=int(cfg.get('distractor_appearance_id', 4)), spline=True)
                            except Exception:
                                try:
                                    ucv.set_appearance(name, id=int(cfg.get('distractor_appearance_id', 4)), spline=False)
                                except Exception:
                                    pass
                except Exception:
                    pass
            except Exception:
                pass

            # 每个 episode 的 Table1 指标累积状态
            state = {
                'ar_sum': 0.0,
                'steps': 0,
                'fail_streak': 0,
                'success': False
            }

            # 与原 follow2 相同的 step 回调（使用真值位姿做视野判定并统计 AR/EL/SR）
            rho_max = 750.0
            theta_max = 45.0
            exp_rho = 250.0
            exp_theta = 0.0

            def step_cb(frame_idx, frame_bgr=None):
                # Ctrl-C 触发后立刻请求结束当前 episode
                if _exit_event.is_set():
                    return False
                try:
                    base = getattr(video.env, 'unwrapped', video.env)
                    ucv = base.unrealcv if hasattr(base, 'unrealcv') else video.env.unrealcv
                    if hasattr(base, 'player_list') and isinstance(base.player_list, (list, tuple)) and len(base.player_list) >= 2:
                        tracker_name = base.player_list[0]
                        target_name  = base.player_list[1]
                    elif hasattr(base, 'player_list') and hasattr(base, 'tracker_id') and hasattr(base, 'target_id'):
                        tracker_name = base.player_list[base.tracker_id]
                        target_name  = base.player_list[base.target_id]
                    elif hasattr(base, 'target_list') and len(base.target_list) >= 2:
                        tracker_name = base.target_list[0]
                        target_name  = base.target_list[1]
                    else:
                        return True
                    tp = ucv.get_obj_pose(tracker_name)
                    gp = ucv.get_obj_pose(target_name)
                except Exception:
                    return True
                dx = gp[0] - tp[0]
                dy = gp[1] - tp[1]
                rho = (dx*dx + dy*dy) ** 0.5
                yaw = tp[4]
                import math as _math
                theta = (_math.degrees(_math.atan2(dy, dx)) - yaw + 180.0) % 360.0 - 180.0
                visible = (rho <= rho_max) and (abs(theta) <= theta_max)
                state['fail_streak'] = 0 if visible else (state['fail_streak'] + 1)
                r = 1.0 - (abs(rho - exp_rho) / rho_max) - (abs(theta - exp_theta) / theta_max)
                state['ar_sum'] += r
                state['steps'] += 1
                if state['fail_streak'] > 50:
                    state['success'] = False
                    return False
                if state['steps'] >= 500:
                    state['success'] = True if state['fail_streak'] == 0 else False
                    return False
                return True

            try:
                res = f.start_mission(device, cfg, tracker, detector, video, vehicle, feature_extractor, step_cb=step_cb)
            except KeyboardInterrupt:
                _cleanup_and_exit(130)
            status = res.get('status') if isinstance(res, dict) else str(res)

            # 汇总本 episode 指标
            ar = float(state['ar_sum'])
            el = int(state['steps'])
            sr = 1.0 if (state['success'] and el >= 500) else 0.0
            local_sum_ar += ar
            local_sum_el += el
            local_sum_sr += sr
            try:
                ts = time.strftime('%Y-%m-%d %H:%M:%S')
                with open(log_path, 'a', encoding='utf-8') as f_out:
                    f_out.write(f"{tag}, {ep+1}, {ar:.4f}, {el}, {sr:.4f}, {status}, {ts}\n")
            except Exception:
                pass
            print(f"[{tag}] Episode {ep+1}/{runs} done. {{'ar': {ar:.2f}, 'el': {el}, 'sr': {sr:.2f}, 'status': '{status}'}}")

            # 确保每个 episode 后刷新输出
            try:
                sys.stdout.flush(); sys.stderr.flush()
            except Exception:
                pass

        if _exit_event.is_set():
            _cleanup_and_exit(130)
        if runs > 0:
            mean_ar = local_sum_ar / runs
            mean_el = local_sum_el // runs
            mean_sr = local_sum_sr / runs
            print(f"[{tag}] Episodes={runs} Mean AR={mean_ar:.2f} Mean EL={mean_el} Mean SR={mean_sr:.2f}")
        # 汇总行
        try:
            ts = time.strftime('%Y-%m-%d %H:%M:%S')
            with open(log_path, 'a', encoding='utf-8') as f_out:
                f_out.write(f"{tag}, mean, {mean_ar:.4f}, {mean_el}, {mean_sr:.4f}, -, {ts}\n")
        except Exception:
            pass
        try:
            sys.stdout.flush(); sys.stderr.flush()
        except Exception:
            pass

    try:
        run_eval_loop('Eval-Table1-Module')
    except KeyboardInterrupt:
        try:
            print('[Eval-Table1] KeyboardInterrupt, terminating...', flush=True)
        except Exception:
            pass
        _cleanup_and_exit(130)

    # 显式清理 UnrealCV 连接，避免挂起
    try:
        base = getattr(video, 'env', None)
        base = getattr(base, 'unwrapped', base)
        if hasattr(base, 'close'):
            try:
                base.close()
            except Exception:
                pass
        if hasattr(video, 'release'):
            try:
                video.release()
            except Exception:
                pass
    except Exception:
        pass


if __name__ == '__main__':
    main()


