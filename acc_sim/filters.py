# acc_sim/filters.py


class KalmanFilter:
    def __init__(self, x0, P0, Q, R):
        self.x = x0
        self.P = P0
        self.Q = Q
        self.R = R

    def predict(self):
        self.P += self.Q

    def update(self, z):
        K = self.P / (self.P + self.R)
        self.x = self.x + K * (z - self.x)
        self.P = (1 - K) * self.P
        return self.x
    
    '''
    def kalman_gain_scalar(P_pred: float, R: float) -> float:
        """
        Scalar Kalman gain: K = P / (P + R)
        """
        denom = P_pred + R
        if denom <= 0:
            raise ValueError("P_pred + R must be > 0.")
        return float(P_pred / denom)
    '''