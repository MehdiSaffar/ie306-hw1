import numpy as np
import numpy.random as npr
import pandas as pd
from functools import partial, wraps
import simpy
import builtins
import itertools as it
import util

SHIFT_DURATION = 8 * 60
BREAK_DURATION = 3
ROOT_SEED = 1
RUN_COUNT = 10


def get_shift_number(minutes):
    return minutes // SHIFT_DURATION


def get_shift_start(shift):
    return shift * SHIFT_DURATION


def lognormal():
    LOGNORMAL_MEAN = np.log10(144/np.sqrt(180))
    LOGNORMAL_SIGMA = np.sqrt(np.log10(180/144))
    return npr.lognormal(mean=LOGNORMAL_MEAN, sigma=LOGNORMAL_SIGMA)


class World:
    def print(self, *args, **kwargs):
        # print(*args, **kwargs)
        pass

    def __init__(self, seed, max_answered_calls):
        npr.seed(seed)
        self.seed = seed
        self.env = simpy.Environment()

        self.operator1 = Operator(
            self.env,
            'operator_1',
            get_service_time=lognormal
        )

        self.operator2 = Operator(
            self.env,
            'operator_2',
            get_service_time=lambda: npr.uniform(1, 7)
        )

        self.avrs = simpy.Resource(self.env, capacity=100)

        # --- stats
        self.max_answered_calls = max_answered_calls

        self.answered_calls = 0
        self.incoming_calls = 0

        self.avrs_time = 0
        self.waiting_time = 0

        self.wrongly_routed_people = 0
        self.renegs = 0

        # -- this is for operator thing
        # self.avg_people_waiting = 0
        # self.people_waiting = 0
        # -- this is for operator thing

        self.ends = self.env.event()

    def run(self):
        self.env.process(self.customer_generator())
        self.env.process(self.break_generator(self.operator1))
        self.env.process(self.break_generator(self.operator2))
        self.env.run(until=self.ends)

    def break_generator(self, chosen_operator):
        for i in it.count():
            yield self.env.timeout(npr.exponential(60))
            Break(f'Break {i}', self.env, self, chosen_operator)

    def customer_generator(self):
        for i in it.count():
            yield self.env.timeout(npr.exponential(6))
            Customer(f'Customer {i}', self.env, self)

    def get_stats(self):
        env = self.env
        now = env.now
        op1, op2 = self.operator1, self.operator2

        system_time = self.waiting_time + self.avrs_time + op1.service_time + op2.service_time
        correctly_routed_people = self.incoming_calls - self.wrongly_routed_people

        return dict(
            seed=self.seed,
            end_time=now,
            shift=get_shift_number(now),
            max_answered_calls=self.max_answered_calls,
            system_time=system_time,
            system_util=self.avrs_time/now,
            op1_util=op1.service_time/now,
            op2_util=op2.service_time/now,
            # waiting_time=self.waiting_time,
            avg_waiting_time=self.waiting_time/correctly_routed_people,
            waiting_to_system_ratio=self.waiting_time/system_time,
            reneg_ratio=self.renegs/self.incoming_calls,
        )

        # print()
        # print(f'Used seed: ', self.seed)
        # print(f'Utilization of the answering system: {system_utilization * 100:.{2}f}%')
        # print(f'Utilization of the operator 1: {operator1_utilization * 100:.{2}f}%')
        # print(f'Utilization of the operator 2: {operator2_utilization * 100:.{2}f}%')

        # # to calculate avg total waiting time I have divided waiting time to people who enters the queue for operators. So, I discarded wrongly routed ones
        # # from all incoming ones because these people never wait. They are irrelevant.
        # print(f'Average total waiting time is {avg_waiting_time:.{2}f} minutes')
        # print(f'Total waiting time/total system time ratio is {ratio * 100:.{2}f}%')
        # print(f'Ratio of reneg people is {reneg_ratio * 100:.{2}f}%')


class Operator(simpy.PriorityResource):
    def __init__(self, env, name,  get_service_time):
        super().__init__(env, capacity=1)
        self.get_service_time = get_service_time
        self.service_time = 0
        self.name = name


class Customer():
    def __init__(self, name, env, world):
        self.env = env
        self.name = name
        self.world = world
        self.action = env.process(self.call())
        self.priority = 0

    def print(self, *args, **kwargs):
        self.world.print(f'{self.name}:', *args, **kwargs)

    def end(self):
        self.chosen_operator.service_time += self.service_duration
        pass

    def call(self):
        self.print(f'initiated a call at {self.env.now}')
        if world.avrs.count == 100:
            self.print(f'AVRS full')
            return

        self.world.incoming_calls += 1
        with world.avrs.request() as req:
            yield req
            self.print(f'is in the avrs at {self.env.now}')
            yield self.env.process(self.introduce_yourself())
            self.print(f'is done avrs at {self.env.now}')

        if self.chosen_operator is None:
            self.print("Wrong operator")
            self.world.wrongly_routed_people += 1
            return

        before_wait = self.env.now
        with self.chosen_operator.request(priority=self.priority) as req:
            self.print(f'is waiting for operator {self.chosen_operator.name} at {before_wait}')

            results = yield req | self.env.timeout(10)
            queue_duration = self.env.now - before_wait
            self.world.waiting_time += queue_duration

            if req not in results:  # reneg
                self.print("Reneging")
                self.world.renegs += 1
                return

            operator_service_time = self.chosen_operator.get_service_time()
            self.print(
                f'meet with operator {self.chosen_operator.name} at {self.env.now} waited operator for {queue_duration} minutes service time will be {operator_service_time}')
            yield self.env.timeout(operator_service_time)
            self.chosen_operator.service_time += operator_service_time
            self.print(f'exiting the system at {self.env.now}')

            # he finished with operator
            self.world.answered_calls += 1
            if self.world.answered_calls == self.world.max_answered_calls:
                self.world.ends.succeed()
                return

    def introduce_yourself(self):
        duration = npr.exponential(5)
        yield self.env.timeout(duration)
        self.world.avrs_time += duration

        self.chosen_operator = self.world.operator1 if npr.rand() <= 0.3 else self.world.operator2
        if npr.rand() <= 0.1:
            self.chosen_operator = None


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
        self.world.print(f'{self.name}:', *args, **kwargs)

    def take_break(self):
        with self.chosen_operator.request(priority=self.priority) as req:
            yield req

            if self.assigned_shift_number != get_shift_number(self.env.now):
                self.print("SUICIDE BOMB")
                return

            next_shift_start_time = get_shift_start(self.assigned_shift_number + 1)

            actual_break_duration = min(next_shift_start_time - self.env.now, BREAK_DURATION)
            self.print(f'{self.chosen_operator.name} break started at {self.env.now}')
            yield self.env.timeout(actual_break_duration)


npr.seed(ROOT_SEED)

seeds = [npr.randint(0, 100) for _ in range(RUN_COUNT)]

all_stats = []
for max_answered_calls in [1000]:
    stats_df = pd.DataFrame()
    for i, seed in enumerate(seeds):
        print(f'running {max_answered_calls} {i} {seed}')
        world = World(seed=seed, max_answered_calls=max_answered_calls)
        world.run()
        stats_df = stats_df.append(dict(run=i, **world.get_stats()), ignore_index=True)

    # stats_df = stats_df[[
    #     'run',
    #     'seed',
    #     'max_answered_calls',
    #     'end_time',
    #     'shift',
    #     'system_util',
    #     'op1_util',
    #     'op2_util',
    #     'avg_waiting_time',
    #     'waiting_to_system_ratio',
    #     'reneg_ratio',
    # ]]

    stats_df = stats_df.round(3)
    stats_df.to_csv(f'./stats.{ROOT_SEED}.{max_answered_calls}.csv', index=False)
    stats_df.describe().to_csv(f'./stats.{ROOT_SEED}.{max_answered_calls}.report.csv')

    all_stats.append(stats_df)

all_stats_df = pd.concat(all_stats)
all_stats_df.to_csv(f'./stats.{ROOT_SEED}.all.csv', index=False)
all_stats_df.describe().to_csv(f'./stats.{ROOT_SEED}.all.report.csv')

for s in all_stats:
    print(stats_df)
