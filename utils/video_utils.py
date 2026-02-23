"""视频读取与日志记录工具"""
from tracking.visualization import save_raw_frame, save_follow_stream_frame


def read_and_log(video, cfg, stage=None, save_images_to=None, **follow_kwargs):
    """读取视频帧并保存日志
    
    Args:
        video: OpenCV VideoCapture 对象
        cfg: 配置字典
        stage: 阶段名称（可选）
        save_images_to: 保存根目录（可选）
        **follow_kwargs: 传递给 save_follow_stream_frame 的参数
        
    Returns:
        (ok, frame): 是否成功，帧图像
    """
    ok, fr = video.read()
    if ok:
        try:
            # 若未显式传入保存目录，则回退到 cfg['save_images_to']
            save_to = save_images_to if save_images_to is not None else cfg.get('save_images_to')
            save_raw_frame(fr, save_images_to=save_to)
        except Exception:
            pass
        if stage is not None:
            try:
                save_follow_stream_frame(cfg, fr, stage=stage, **follow_kwargs)
            except Exception:
                pass
    return ok, fr




