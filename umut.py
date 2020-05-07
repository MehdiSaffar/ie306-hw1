import numpy as np
from functools import partial, wraps
import simpy
import builtins
import itertools as it


def patch_resource(resource, pre=None, post=None):
    """Patch *resource* so that it calls the callable *pre* before each
    put/get/request/release operation and the callable *post* after each
    operation.  The only argument to these functions is the resource
    instance.
    """
    def get_wrapper(func):
        # Generate a wrapper for put/get/request/release
        @wraps(func)
        def wrapper(*args, **kwargs):
            # This is the actual wrapper
            # Call "pre" callback
            if pre:
                pre(resource)

            # Perform actual operation
            ret = func(*args, **kwargs)

            # Call "post" callback
            if post:
                post(resource)

            return ret
        return wrapper

    # Replace the original operations with our wrapper
    for name in ['put', 'get', 'request', 'release']:
        if hasattr(resource, name):
            setattr(resource, name, get_wrapper(getattr(resource, name)))


class World:
    def __init__(self, env):
        # Init operator 1 with lognormal dist
        self.operator1 = Operator(env, 'opeartor_1', get_service_time=lambda: np.random.lognormal(mean=LOGNORMAL_MEAN, sigma=LOGNORMAL_SIGMA))

        # Init operator 2 with random uniform dist
        self.operator2 = Operator(env, 'operator_2', get_service_time=lambda: np.random.uniform(low=1, high=7))

        self.speech_recognition_operator = simpy.Resource(env, capacity=100)
        self.answered_call_count = 0
        self.total_incoming_calls = 0
        self.max_answered_call_count = 5000
        self.avrs_time = 0
        self.total_waiting_time = 0
        self.wrongly_routed_people = 0
        self.unsatisfied_people = 0

        # -- this is for operator thing
        self.avg_people_waiting = 0
        self.total_people_waiting = 0
        # -- this is for operator thing

        self.ends = env.event()


class Operator(simpy.PriorityResource):
    def __init__(self, env, name,  get_service_time):
        super().__init__(env, capacity=1)
        self.get_service_time = get_service_time
        self.total_service_time = 0
        self.name = name


class Customer():
    def __init__(self, name, env, world):
        self.env = env
        self.name = name
        self.world = world
        self.action = env.process(self.call())
        self.priority = 0

        # self.arrival_time = env.now
        # self.queue_time = 0
        # self.service_time = 0

    def print(self, *args, **kwargs):
        print(f'{self.name}:', *args, **kwargs)

    def call(self):
        self.print(f'initiated a call at {self.env.now}')
        if world.speech_recognition_operator.count == 100:
            self.print(f'AVRS full')
            return

        self.world.total_incoming_calls += 1
        with world.speech_recognition_operator.request() as req:
            yield req
            self.print(f'is in the speech recognition system at {self.env.now}')

            # self.queue_time += self.env.now - self.arrival_time
            yield self.env.process(self.introduce_yourself())
            self.print(f'is done speech recognition at {self.env.now}')

        if self.chosen_operator is None:
            self.print("Wrong operator")
            # rejected
            return

        with self.chosen_operator.request(priority=self.priority) as req:
            before_wait = env.now
            self.print(f'is started waiting for operator {self.chosen_operator.name} at {before_wait}')
            results = yield req | self.env.timeout(10)
            after_wait = env.now
            wait_duration = after_wait - before_wait
            self.world.total_waiting_time += wait_duration


            if req not in results:  # continue because of timeout
                self.print("Reneging")
                self.world.unsatisfied_people += 1
                return

            service_time = self.chosen_operator.get_service_time()
            self.print(f'meet with operator {self.chosen_operator.name} at {after_wait} waited operator for {wait_duration} minutes service time will be {service_time}')
            yield self.env.timeout(service_time)
            self.chosen_operator.total_service_time += service_time
            self.print(f'exiting the system at {env.now}')

            # he finished with operator
            self.world.answered_call_count += 1
            if self.world.answered_call_count == self.world.max_answered_call_count:
                self.world.ends.succeed()

            # continue because request given

    def introduce_yourself(self):
        duration = np.random.exponential(5)
        yield self.env.timeout(duration)
        self.world.avrs_time += duration

        self.chosen_operator = self.world.operator1 if np.random.rand() <= 0.3 else self.world.operator2
        if np.random.rand() <= 0.1:
            self.chosen_operator = None
            self.world.unsatisfied_people += 1
            self.world.wrongly_routed_people += 1
        # self.service_time += duration


SHIFT_DURATION = 8 * 60
BREAK_DURATION = 3
SEED = 547885

LOGNORMAL_MEAN = np.log10(144/np.sqrt(180))
LOGNORMAL_SIGMA = np.sqrt(np.log10(180/144))


def get_shift_number(minutes):
    return minutes // SHIFT_DURATION


def get_shift_start(shift):
    return shift * SHIFT_DURATION


class Break():
    def __init__(self, name, env, world, chosen_operator):
        self.env = env
        self.world = world
        self.name = name
        self.action = env.process(self.take_break())
        self.chosen_operator = chosen_operator
        self.assigned_shift_number = get_shift_number(env.now)
        self.priority = 1

    def print(self, *args, **kwargs):
        print(f'{self.name}:', *args, **kwargs)

    def take_break(self):
        with self.chosen_operator.request(priority=self.priority) as req:
            yield req

            if self.assigned_shift_number != get_shift_number(env.now):
                self.print("SUICIDE BOMB")
                return

            next_shift_start_time = get_shift_start(self.assigned_shift_number + 1)

            actual_break_duration = min(next_shift_start_time - env.now, BREAK_DURATION)
            self.print(f'{self.chosen_operator.name} break started at {env.now}')
            yield self.env.timeout(actual_break_duration)


def break_generator(env, world, chosen_operator):
    for i in it.count():
        yield env.timeout(np.random.exponential(60))
        Break(f'Break {i}', env, world, chosen_operator)


def customer_generator(env, world):
    """Generate new customers that arrive into system."""
    for i in it.count():
        yield env.timeout(np.random.exponential(6))
        Customer(f'Customer {i}', env, world)


np.random.seed(SEED)
env = simpy.Environment()
world = World(env)
env.process(customer_generator(env, world))
env.process(break_generator(env, world, world.operator1))
env.process(break_generator(env, world, world.operator2))
env.run(until=world.ends)

total_system_time = world.total_waiting_time + world.avrs_time + world.operator1.total_service_time + world.operator2.total_service_time


print("DONE", world.answered_call_count, get_shift_number(env.now))

print(f'\nUtilization of the answering system: {world.avrs_time/(env.now*100)*100} %')
print(f'Utilization of the operator 1: {world.operator1.total_service_time/env.now*100} %')
print(f'Utilization of the operator 2: {world.operator2.total_service_time/env.now*100} %')
#to calculate avg total waiting time I have divided total_waiting time to people who enters the queue for operators. So, I discarded wrongly routed ones 
#from all incoming ones because these people never wait. They are irrelevant. 
print(f'Average total waiting time is {world.total_waiting_time/(world.total_incoming_calls - world.wrongly_routed_people)}')
print(f'Total waiting time/total system time ratio is {world.total_waiting_time/total_system_time}')
print(f'Ratio of unsatisfied people is {world.unsatisfied_people/world.total_incoming_calls}')
