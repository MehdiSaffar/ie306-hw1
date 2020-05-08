import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
import pandas as pd
import simpy
import itertools as it

SHIFT_DURATION = 8 * 60
BREAK_DURATION = 3
ROOT_SEED = 1
RUN_COUNT = 10

# events = []

def get_shift_number(minutes):
    """
    Returns the shift index 
    """
    return minutes // SHIFT_DURATION


def get_shift_start(shift):
    """
    Returns the first minute of a shift
    """
    return shift * SHIFT_DURATION


def lognormal():
    LOGNORMAL_MEAN = np.log(144/np.sqrt(180))
    LOGNORMAL_SIGMA = np.sqrt(np.log(180/144))
    return npr.lognormal(mean=LOGNORMAL_MEAN, sigma=LOGNORMAL_SIGMA)


class World:
    """
    World class that hold information about simulation miniworld and provides necessary functionalities
    """

    # In order to see output uncomment print statement
    def print(self, *args, **kwargs):
        # print(*args, **kwargs)
        pass

    def __init__(self, seed, max_answered_calls):
        # Set seed and max answered calls limit
        npr.seed(seed)
        self.seed = seed
        self.env = simpy.Environment()

        self.operator1 = Operator(  # Create first operator
            self.env,
            'operator_1',
            get_service_time=lognormal  # Service time has lognormal distribution
        )

        self.operator2 = Operator(  # Create second operator
            self.env,
            'operator_2',
            get_service_time=lambda: npr.uniform(1, 7)  # Service time has uniform distribution
        )

        self.avrs = simpy.Resource(self.env, capacity=100)  # Create avrs resource

        # --- stats
        self.max_answered_calls = max_answered_calls

        # Answered call is a call makes it all the way to the end
        self.answered_calls = 0

        # All calls that are not rejected by AVRS
        self.incoming_calls = 0

        # Sum of time inside AVRS for each customer
        self.avrs_time = 0

        # Sum of waiting times of each customer
        self.waiting_time = 0

        # Count wrongly routed people
        self.wrongly_routed_people = 0

        # Count reneged people
        self.renegs = 0

        # Create simulation end event
        self.ends = self.env.event()

    def run(self):
        """ 
        Run the simulation until ends
        """
        self.env.process(self.customer_generator())
        self.env.process(self.break_generator(self.operator1))
        self.env.process(self.break_generator(self.operator2))
        self.env.run(until=self.ends)

    def break_generator(self, chosen_operator):
        """ 
        Generate operator breaks
        """
        for i in it.count():  # Create breaks until ends event
            yield self.env.timeout(npr.exponential(60))  # Break generation interval has exponential distribution
            Break(f'Break {i}', self.env, self, chosen_operator)  # Generate actual Break object

    def customer_generator(self):
        """
        Generate customers
        """
        for i in it.count():  # Generate customer until ends event
            yield self.env.timeout(npr.exponential(6))  # Customer generation interval has exponential distribution
            Customer(f'Customer {i}', self.env, self)  # Generate actual Customer object

    def get_stats(self):
        """
        Get world statistics
        """
        env = self.env
        now = env.now
        op1, op2 = self.operator1, self.operator2

        # System time = waiting time + avrs time + operator service times
        system_time = self.waiting_time + self.avrs_time + op1.service_time + op2.service_time

        # Calculate correctly routed people
        correctly_routed_people = self.incoming_calls - self.wrongly_routed_people

        # Get average number of customers waiting in operator #1 queue
        avg_number_of_customers_1 = op1.avg_number_of_customers()

        # Get average number of customers waiting in operator #2 queue
        avg_number_of_customers_2 = op2.avg_number_of_customers()

        return dict(  # Create a dictionary with the gathered statistics data
            seed=self.seed,
            max_answered_calls=self.max_answered_calls,

            system_util=self.avrs_time/(100*now),
            op1_utilization=op1.service_time/now,
            op2_utilization=op2.service_time/now,
            avg_waiting_time=self.waiting_time/correctly_routed_people,
            waiting_to_system_ratio=self.waiting_time/system_time,
            avg_number_of_waiting_customers_op_1=avg_number_of_customers_1,
            avg_number_of_waiting_customers_op_2=avg_number_of_customers_2,
            unsatisfied_people=self.renegs+self.wrongly_routed_people

            # end_time=now,
            # shift=get_shift_number(now),
            # system_time=system_time,
            # waiting_time=self.waiting_time,
        )


class Operator(simpy.PriorityResource):
    def __init__(self, env, name,  get_service_time):
        super().__init__(env, capacity=1)
        self.env = env
        self.get_service_time = get_service_time
        self.service_time = 0
        self.name = name
        self.queue_duration = 0  #  Total duration of people waiting in operator queue

    def avg_number_of_customers(self):
        return self.queue_duration / self.env.now  #  Avg number of people in operator queue

    # this was used to generate a plot of people waiting in queue over tiem
    # def request(self, priority=0, preempt=True):
    #     events.append({(self.name): len(self.queue), 'time': self.env.now})
    #     return super().request(priority, preempt)

    # def release(self, request):
    #     events.append({(self.name): len(self.queue), 'time': self.env.now})
    #     return super().release(request)


class Customer():
    def __init__(self, name, env, world):
        self.env = env
        self.name = name
        self.world = world
        self.action = env.process(self.call())
        self.priority = 0

    def print(self, *args, **kwargs):
        self.world.print(f'{self.name}:', *args, **kwargs)

    def call(self):
        self.print(f'initiated a call at {self.env.now}')
        # if all channels of the system is full immediately exit system.
        if world.avrs.count == 100:
            self.print(f'AVRS full')
            return

        # increment incoming calls and request one of the empty automated answering systems.
        self.world.incoming_calls += 1
        with world.avrs.request() as req:
            yield req
            self.print(f'is in the avrs at {self.env.now}')
            yield self.env.process(self.introduce_yourself())
            self.print(f'is done avrs at {self.env.now}')

        # if operator is set to none it means that customer is wrongly routed. So, I exits<the system.
        if self.chosen_operator is None:
            self.print("Wrong operator")
            self.world.wrongly_routed_people += 1
            return

        before_wait = self.env.now

        # after answering system user enters into queue of operator that he has been routed to.
        with self.chosen_operator.request(priority=self.priority) as req:
            self.print(f'is waiting for operator {self.chosen_operator.name} at {before_wait}')
            # check if operator is assigned to user before 10 minutes of wait.
            results = yield req | self.env.timeout(10)
            queue_duration = self.env.now - before_wait
            self.world.waiting_time += queue_duration
            self.chosen_operator.queue_duration += queue_duration

            if req not in results:  # if 10 mins are passed then leave.
                self.print("Reneging")
                self.world.renegs += 1
                return

            # get the service time according to expected distribution
            operator_service_time = self.chosen_operator.get_service_time()
            self.print(
                f'meet with operator {self.chosen_operator.name} at {self.env.now} waited operator for {queue_duration} minutes service time will be {operator_service_time}')
            yield self.env.timeout(operator_service_time)  # wait service of operator to end.
            self.chosen_operator.service_time += operator_service_time  # increase operator's service time.
            self.print(f'exiting the system at {self.env.now}')
            # At that point call is answered  by an operator successfully. Increment answered call counter.
            self.world.answered_calls += 1
            # If answered calls are reached maximum calls of simulation, terminate.
            if self.world.answered_calls == self.world.max_answered_calls:
                self.world.ends.succeed()
                return

    # This function is called when a user enters the answering system.
    def introduce_yourself(self):
        duration = npr.exponential(5)  # Duration of automated answering is calculated.
        yield self.env.timeout(duration)
        self.world.avrs_time += duration
        self.chosen_operator = self.world.operator1 if npr.rand() <= 0.3 else self.world.operator2  # routing to operator
        if npr.rand() <= 0.1:
            self.chosen_operator = None  # wrong routing case


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

    # When its time comes a take_break is called to make operator to take a break
    def take_break(self):
        # Break requests te operator but it has a lower priority than customers._
        # So, if we have a customer waiting in the system break cannot start
        with self.chosen_operator.request(priority=self.priority) as req:
            yield req
            # if break is created for previous shift, then discard break
            if self.assigned_shift_number != get_shift_number(self.env.now):
                self.print("Break discarded")
                return

            next_shift_start_time = get_shift_start(self.assigned_shift_number + 1)

            actual_break_duration = min(next_shift_start_time - self.env.now, BREAK_DURATION)
            self.print(f'{self.chosen_operator.name} break started at {self.env.now}')
            yield self.env.timeout(actual_break_duration)


npr.seed(ROOT_SEED)  # Set root seed

seeds = [npr.randint(0, 100) for _ in range(RUN_COUNT)]  # Generate a random seed for each run

all_stats = []
for max_answered_calls in [1000, 5000]:
    stats_df = pd.DataFrame()
    for i, seed in enumerate(seeds):
        print(f'running {max_answered_calls} {i} {seed}')
        world = World(seed=seed, max_answered_calls=max_answered_calls)
        world.run()
        stats_df = stats_df.append(dict(run=i, **world.get_stats()), ignore_index=True)

    stats_df = stats_df.round(3)

    old_cols = ['run', 'seed', 'system_util', 'op1_utilization', 'op2_utilization', 'avg_waiting_time', 'waiting_to_system_ratio',
                'avg_number_of_waiting_customers_op_1', 'avg_number_of_waiting_customers_op_2', 'unsatisfied_people']
    new_cols = ['Run', 'Seed', 'Sys. Util', 'Op1 Util', 'Op2 Util', 'Avg. Wait',
                'W/S', 'Avg. \# in Op1 Q', 'Avg. # in Op2 Q', '# Unsatisfied']

    print_df = stats_df[old_cols]
    print_df.columns = new_cols

    print_df.to_csv(f'./stats.{ROOT_SEED}.{max_answered_calls}.csv', index=False)
    print_df.describe().round(3).to_csv(f'./stats.{ROOT_SEED}.{max_answered_calls}.report.csv')

    all_stats.append(stats_df)

all_stats_df = pd.concat(all_stats)

print_df = all_stats_df[old_cols + ['max_answered_calls']]
print_df.columns = new_cols + ['Max Answered Calls']
print_df.to_csv(f'./stats.{ROOT_SEED}.all.csv', index=False)
print_df.describe().round(3).to_csv(f'./stats.{ROOT_SEED}.all.report.csv')

for s in all_stats:
    print(stats_df)

# This was used to plot # of people in queue of operators
# events_df = pd.DataFrame(events)

# events_df.plot()
# events_df['y'] = events_df.event.cumsum()
# events_df.fillna(0)
# plt.subplot(211)
# plt.plot(events_df['time'], events_df['operator_1'], drawstyle='steps')
# plt.subplot(212)
# plt.plot(events_df['time'], events_df['operator_2'], drawstyle='steps')
# plt.show()
# # plt.plot(events_df)

# print(events_df)
