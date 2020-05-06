import simpy as sp
import numpy as np

class Globals:
    interarrival_time_mean = 6

env = sp.Environment()

class Operator():
    def __init__(self):
        super().__init__()
        # self.operator_num = 1 yada 2

class Customer(sp.Process):
    def __init__(self, env):
        super().__init__(env, self.generate)
        self.is_satisfied = False
        self.waiting_time = 0
        self.service_time = 0
    
    def generate(self):
        pass


class AVRS():
    def __init__(self):
        self.recognition_time = np.random.exponential(scale=5)

    ## --- 
    # customer  = #...
    # operator_val = np.random.uniform()
    # yield put_to_queue, customer, 1 if operator_val > 0.3 else 2
    ## --- 


def cust(env: sp.Environment):
    while True:
        yield env.timeout(1)

def arrival(env: sp.Environment):
    while True:
        interarrival_time = np.random.exponential(scale=Globals.interarrival_time_mean)
        # customer = Customer(env)
        yield env.timeout(interarrival_time)
        yield env.process(cust(env))
        # yield customer
        print(f"new customer {env.now}, interarrival time {interarrival_time}")

env.process(arrival(env))
env.run(until=24*60)