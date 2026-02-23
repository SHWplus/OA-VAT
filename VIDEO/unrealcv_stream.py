import cv2
import gym
import numpy as np
import gym_unrealcv  # noqa: F401
from gym_unrealcv.envs.wrappers import configUE


class UnrealCVStream:
    def __init__(self, env_id, width=320, height=240):
        # Some gym versions do not accept kwargs in make(); create with ID only
        self.env = gym.make(env_id)
        # Enforce offscreen rendering and set native resolution BEFORE first reset
        try:
            self.env = configUE.ConfigUEWrapper(self.env, offscreen_rendering=True)
        except Exception:
            pass
        try:
            base = getattr(self.env, 'unwrapped', self.env)
            base.resolution = (width, height)
            base.offscreen_rendering = True
        except Exception:
            pass
        self.env.reset()
        self.width = width
        self.height = height
        self.tracker_idx = getattr(self.env, 'tracker_id', 0)
        self._stop_discrete = None
        try:
            plist = self.env.player_list
            tracker_name = plist[self.tracker_idx]
            alist = self.env.agents[tracker_name]['discrete_action']
            for i, a in enumerate(alist):
                if len(a) >= 2 and a[0] == 0 and a[1] == 0:
                    self._stop_discrete = i
                    break
        except Exception:
            pass

    def read(self):
        try:
            actions = None
            try:
                n = len(self.env.player_list)
                actions = [None] * n
                if getattr(self.env, 'action_type', 'Discrete') == 'Discrete':
                    actions[self.tracker_idx] = self._stop_discrete if self._stop_discrete is not None else 0
                else:
                    actions[self.tracker_idx] = [0.0, 0.0]
            except Exception:
                pass
            if actions is not None:
                # Some envs expect a list the same length as player_list; fall back to noop on mismatch
                try:
                    self.env.step(actions)
                except Exception:
                    try:
                        self.env.step([None])
                    except Exception:
                        pass
            # Prefer UnrealCV direct color read to get native resolution
            frame_arr = None
            try:
                base = getattr(self.env, 'unwrapped', self.env)
                ucv = base.unrealcv if hasattr(base, 'unrealcv') else None
                if ucv is not None:
                    if hasattr(base, 'cam_list') and hasattr(base, 'tracker_id'):
                        cam_id = base.cam_list[base.tracker_id]
                    elif hasattr(base, 'cam_id') and isinstance(base.cam_id, (list, tuple)):
                        cam_id = base.cam_id[0]
                    else:
                        cam_id = 1
                    frame_arr = ucv.read_image(cam_id, 'color', 'direct')
            except Exception:
                frame_arr = None
            if frame_arr is None:
                frame_arr = self.env.render(mode='rgb_array')
            if frame_arr is None:
                return False, None
            # Only downscale (avoid upscaling blur)
            h, w = frame_arr.shape[:2]
            if w > self.width or h > self.height:
                try:
                    frame_arr = cv2.resize(frame_arr, (self.width, self.height), interpolation=cv2.INTER_AREA)
                except Exception:
                    pass
            return True, frame_arr
        except Exception:
            return False, None

    @property
    def fps(self):
        return 0

    def release(self):
        try:
            self.env.close()
        except Exception:
            pass


