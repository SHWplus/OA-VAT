import numpy as np

class UnrealSimController:
    def __init__(self, env, tracker_index=0, max_speed=60):
        # 外层 env（含 TimeLimit 等 wrappers）
        self.env = env
        # 最内层 env，包含 target_list / agents 等元数据
        self.base = getattr(env, 'unwrapped', env)
        self.tracker_index = tracker_index
        self.max_speed = int(max(1, min(200, max_speed)))
        # 若 1v1 环境则用 0 作为 tracker 索引
        if hasattr(self.base, 'target_list'):
            self.tracker_index = 0
        elif hasattr(self.base, 'tracker_id'):
            self.tracker_index = getattr(self.base, 'tracker_id', 0)

    def _n_players(self):
        if hasattr(self.base, 'player_list'):
            return len(self.base.player_list)
        if hasattr(self.base, 'target_list'):
            return 2
        return 1

    def _is_drone_tracker(self):
        try:
            if hasattr(self.base, 'player_list'):
                name = self.base.player_list[self.tracker_index]
                info = self.base.agents.get(name, {})
                return str(info.get('agent_type', '')).lower() == 'drone'
        except Exception:
            pass
        return False

    def send_velocity(self, vx, vy, vz, yaw_rate):
        # 裁剪速度
        vx = int(np.clip(vx, -self.max_speed, self.max_speed))
        vy = int(np.clip(vy, -self.max_speed, self.max_speed))
        vz = int(np.clip(vz, -self.max_speed, self.max_speed))
        yaw_rate = int(np.clip(yaw_rate, -self.max_speed, self.max_speed))

        try:
            n = self._n_players()
            actions = []
            is_drone = self._is_drone_tracker()
            # 1) 先为所有体填充占位动作，保证长度/形状正确
            for _ in range(n):
                if is_drone:
                    actions.append([0, 0, 0, 0])          # [vx, vy, vz, vyaw]
                else:
                    actions.append([0, 0])                # 连续 2 参

            # 2) 写入 tracker 的动作
            if is_drone:
                # 3D 体 4 参
                actions[self.tracker_index] = [vx, vy, vz, yaw_rate]
            else:
                # 1v1/一般平面体 连续 2 参，注意顺序为 (velocity, angle)
                actions[self.tracker_index] = [vy, yaw_rate]

            # 3) 通过外层 env.step(actions) 执行，保持 wrappers 生效
            self.env.step(actions)
        except Exception as e:
            print(f"UnrealSimController send_velocity error: {e}")

    def get_battery(self):
        return 100

    def land(self):
        pass

    def close(self):
        try:
            self.env.close()
        except Exception:
            pass



