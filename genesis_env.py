import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

class GenesisTradingEnv(gym.Env):
    """Entorno de Trading Personalizado de OpenAI Gym para GENESIS IA."""
    metadata = {'render.modes': ['human']}

    def __init__(self, df, initial_balance=1000.0, com_pct=0.0025, reinv_pct=0.20):
        super(GenesisTradingEnv, self).__init__()
        
        # Datos del mercado (Features) - Deben ser numéricos
        self.df = df.reset_index(drop=True)
        # Seleccionamos las columnas que la IA usará para tomar decisiones
        self.features = ['Open', 'High', 'Low', 'Close', 'Volume', 'RSI', 'ADX', 'ATR', 'Z_Score']
        
        # Parámetros del entorno
        self.initial_balance = initial_balance
        self.com_pct = com_pct
        self.reinv_pct = reinv_pct
        
        # Estado actual
        self.current_step = 0
        self.balance = self.initial_balance
        self.shares_held = 0
        self.average_price = 0
        
        # Espacio de Acciones (Action Space): 0 = Mantener, 1 = Comprar, 2 = Vender
        self.action_space = spaces.Discrete(3)
        
        # Espacio de Observación (Observation Space): Matriz con las features de la vela actual
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(len(self.features),), dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.balance = self.initial_balance
        self.shares_held = 0
        self.average_price = 0
        
        return self._next_observation(), {}

    def _next_observation(self):
        # Devuelve las features de la vela actual
        obs = self.df.loc[self.current_step, self.features].values
        return obs.astype(np.float32)

    def step(self, action):
        # 1. Ejecutar Acción
        current_price = self.df.loc[self.current_step, 'Close']
        reward = 0
        
        if action == 1: # Comprar
            if self.balance > 0 and self.shares_held == 0: # Solo compramos si no estamos en posición
                invest_amt = self.balance * self.reinv_pct
                comm_in = invest_amt * self.com_pct
                pos_size = invest_amt - comm_in
                self.shares_held = pos_size / current_price
                self.average_price = current_price
                
        elif action == 2: # Vender
            if self.shares_held > 0:
                gross = self.shares_held * current_price
                comm_out = gross * self.com_pct
                net = gross - comm_out
                
                # Calcular la ganancia de esta operación
                profit = net - (self.shares_held * self.average_price)
                self.balance += profit
                self.shares_held = 0
                
                # La recompensa es el porcentaje de ganancia neta
                reward = profit / (self.shares_held * self.average_price) if (self.shares_held * self.average_price) > 0 else 0

        # 2. Avanzar el tiempo
        self.current_step += 1
        
        # 3. Verificar si terminamos
        terminated = self.current_step >= len(self.df) - 1
        
        # Penalización si nos quedamos sin dinero (aunque con el reinv_pct es difícil llegar a 0)
        if self.balance <= 0:
            terminated = True
            reward = -1 

        obs = self._next_observation()
        
        # Guardamos información extra para el dashboard
        info = {
            'balance': self.balance,
            'portfolio_value': self.balance + (self.shares_held * current_price if self.shares_held > 0 else 0)
        }
        
        return obs, reward, terminated, False, info

    def render(self, mode='human'):
        # Aquí podríamos imprimir el estado en consola para debug
        pass
