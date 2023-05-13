import numpy as np


class AdamOptim():
    def __init__(self, lr=0.04, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.m_dw, self.v_dw = 0, 0
        self.m_db, self.v_db = 0, 0
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.lr = lr

    def update(self, w, b, dw, db, t=1):
        # dw, db are from current minibatch
        # momentum beta 1
        # *** weights *** #
        self.m_dw = self.beta1 * self.m_dw + (1 - self.beta1) * dw
        # *** biases *** #
        self.m_db = self.beta1 * self.m_db + (1 - self.beta1) * db

        # rms beta 2
        # *** weights *** #
        self.v_dw = self.beta2 * self.v_dw + (1 - self.beta2) * (dw**2)
        # *** biases *** #
        self.v_db = self.beta2 * self.v_db + (1 - self.beta2) * (db**2)
        # bias correction
        m_dw_corr = self.m_dw / (1 - self.beta1**t)
        m_db_corr = self.m_db / (1 - self.beta1**t)
        v_dw_corr = self.v_dw / (1 - self.beta2**t)
        v_db_corr = self.v_db / (1 - self.beta2**t)
        # how much weights and biases should be adjusted
        dw = self.lr * (m_dw_corr / (np.sqrt(v_dw_corr) + self.epsilon))
        db = self.lr * (m_db_corr / (np.sqrt(v_db_corr) + self.epsilon))
        return dw, db
