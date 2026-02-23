"""PID 控制器实现"""


class PIDController:
    """PID 控制器类
    
    实现标准的 PID (比例-积分-微分) 控制器
    """
    
    def __init__(self, Kp, Ki, Kd):
        """初始化PID控制器
        
        Args:
            Kp: 比例增益
            Ki: 积分增益
            Kd: 微分增益
        """
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.integral = 0
        self.prev_error = 0

    def update(self, error, dt):
        """更新PID控制器并计算输出
        
        Args:
            error: 当前误差
            dt: 时间步长
            
        Returns:
            float: PID输出值
        """
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt
        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        self.prev_error = error
        return output

    def reset(self):
        """重置PID控制器状态"""
        self.integral = 0
        self.prev_error = 0




