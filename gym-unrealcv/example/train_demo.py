# 屏蔽 Gym 警告（可选）
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import gym
import gym_unrealcv
import time

# 1. 加载环境（端口已更换或释放）
env = gym.make(id='UnrealSearch-RealisticRoomDoor-DiscreteColor-v0')

# 2. 初始化环境（此时观测分辨率应改为 480x640，需看 JSON 配置）
obs = env.reset()
print(f"环境初始化成功！观测图像形状：{obs.shape}")  # 目标输出 (480, 640, 3)

# 3. 仅调用 render()，不传递任何参数
env.render()  # 关键：删除 width/height，依赖 UE4 配置

# 4. 暂停 30 秒查看窗口
print("暂停 30 秒，查看 RealisticRoom 场景窗口...")
time.sleep(30)

# 5. 测试动作（验证窗口稳定性）
for step in range(5):
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    print(f"第 {step+1} 步：动作={action}，奖励={reward:.2f}")
    if done:
        obs = env.reset()

# 6. 关闭环境
env.close()
