import numpy as np
from functools import partial, wraps
import simpy
import builtins
import itertools as it

service_times = []  # Represents service times
queue_wait_times = []  # Represents queue waiting times


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
    def on_op_request(self, resource):
        # Not completely sure about this
        self.total_people_waiting += 1
        self.avg_people_waiting += 1/self.total_people_waiting
        pass

    def on_op_used(self, resource):
        # Not completely sure about this
        self.total_people_waiting -= 1
        self.avg_people_waiting -= 1/self.total_people_waiting
        pass

    def __init__(self, env):
        # Init operator 1 with lognormal dist
        self.operator1 = patch_resource(
            Operator(env, get_service_time=lambda: np.random.lognormal(
                mean=12, sigma=6)),
            pre=self.on_op_request,
            post=self.on_op_used,
        )

        # Init operator 2 with random uniform dist
        self.operator2 = patch_resource(
            Operator(env, get_service_time=lambda: np.random.uniform(low=1, high=7)),
            pre=self.on_op_request,
            post=self.on_op_used,
        )

        self.speech_recognition_operator = simpy.Resource(env, capacity=100)
        self.answered_call_count = 0
        self.max_answered_call_count = 1000
        self.avrs_time = 0
        self.total_waiting_time = 0
        self.unsatisfied_people = 0

        ## -- this is for operator thing
        self.avg_people_waiting = 0
        self.total_people_waiting = 0
        ## -- this is for operator thing

        self.ends = env.event()


class Operator(simpy.PriorityResource):
    def __init__(self, env, get_service_time,):
        super().__init__(env, capacity=1)
        self.get_service_time = get_service_time
        self.total_service_duration = 0


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

        with world.speech_recognition_operator.request() as req:
            yield req
            self.print(f'is assigned to an operator at {self.env.now}')

            # self.queue_time += self.env.now - self.arrival_time
            yield self.env.process(self.introduce_yourself())
            self.print(f'is done at {self.env.now}')

        if self.chosen_operator is None:
            self.print("Wrong operator")
            # rejected
            return

        with self.chosen_operator.request(priority=self.priority) as req:
            before_wait = env.now
            results = yield req | self.env.timeout(10)
            after_wait = env.now
            wait_duration = after_wait - before_wait

            self.world.total_waiting_time += wait_duration


            if req not in results:  # continue because of timeout
                self.print("Reneging")
                self.world.unsatisfied_people += 1
                return

            service_time = self.chosen_operator.get_service_time()
            yield self.env.timeout(service_time)
            self.chosen_operator.total_service_time += service_time

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
            self.world.unsatisfied_people +=1
        # self.service_time += duration


SHIFT_DURATION = 8 * 60
BREAK_DURATION = 3
SEED = 0


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
            self.print("Break started")
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

print("DONE", world.answered_call_count, get_shift_number(env.now))
