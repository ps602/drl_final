"""
Traffic network simulator w/ defined sumo files
@author: Tianshu Chu
"""
import logging
import numpy as np
import pandas as pd
import subprocess
from sumolib import checkBinary
import time
import traci
import xml.etree.cElementTree as ET  # xml accessing api

DEFAULT_PORT = 8000
SEC_IN_MS = 1000

# hard code real-net reward norm
REALNET_REWARD_NORM = 20


class PhaseSet:

    # class to control phase

    def __init__(self, phases):

        #  init method
        self.num_phase = len(phases)  # number of phases
        self.num_lane = len(phases[0])  # number of lanes
        self.phases = phases  # phases array with number of lanes in phase[0] and phase seq later
        # self._init_phase_set()

    @staticmethod  # utility type method to be used on object of the class
    def _get_phase_lanes(phase, signal='r'):
        phase_lanes = []  # declaring array
        for i, l in enumerate(phase):  # if l1 = "rrgr",print(list(enumerate(l1))= [(0,'r'),(1,'r'),(2,'g'),(3,'r')]
            if l == signal:  # checking if phase is r
                phase_lanes.append(i)  # appending the lane with r signal into phase_lanes array
        return phase_lanes
    # basically a method to get the lane numbers with a given signal

    def _init_phase_set(self):
        self.red_lanes = []  # creating an array called red_lanes
        # self.green_lanes = []
        for phase in self.phases:  # iterating on the phases array in init
            self.red_lanes.append(self._get_phase_lanes(phase))  # from the phase array we store the lanes in red
            # self.green_lanes.append(self._get_phase_lanes(phase, signal='G'))


class PhaseMap:
    def __init__(self):
        self.phases = {}  # empty dictionary of the name phases

    def get_phase(self, phase_id, action):
        # phase_type is either green or yellow
        return self.phases[phase_id].phases[int(action)]  # accessing phase_id and action from phases dictionary?

    def get_phase_num(self, phase_id):
        return self.phases[phase_id].num_phase  # corresponding to phase_id we get phase_num?

    def get_lane_num(self, phase_id):
        # the lane number is link number
        return self.phases[phase_id].num_lane  # corresponding to phase_id we get lane_num?

    def get_red_lanes(self, phase_id, action):
        # the lane number is link number
        return self.phases[phase_id].red_lanes[int(action)]  # corresponding to phase_id we get red_lanes?
# basically a class for getting info about the traffic env

class Node:
    def __init__(self, name, neighbor=[], control=False):
        self.control = control  # disabled ??
        # self.edges_in = []  # for reward
        self.lanes_in = []
        self.ilds_in = []  # for state
        self.fingerprint = []  # local policy
        self.name = name
        self.neighbor = neighbor
        self.num_state = 0  # wave and wait should have the same dim
        self.num_fingerprint = 0
        self.wave_state = []  # local state
        self.wait_state = []  # local state
        # self.waits = [] 
        self.phase_id = -1
        self.n_a = 0
        self.prev_action = -1

# defines properties of a node
class TrafficSimulator:
    def __init__(self, config, output_path, is_record, record_stats, port=0):
        self.name = config.get('scenario')
        self.seed = config.getint('seed')
        self.control_interval_sec = config.getint('control_interval_sec')
        self.yellow_interval_sec = config.getint('yellow_interval_sec')
        self.episode_length_sec = config.getint('episode_length_sec')
        self.T = np.ceil(self.episode_length_sec / self.control_interval_sec)
        self.port = DEFAULT_PORT + port
        self.sim_thread = port
        self.obj = config.get('objective')
        self.data_path = config.get('data_path')
        self.agent = config.get('agent')
        self.coop_gamma = config.getfloat('coop_gamma')
        self.cur_episode = 0
        self.norms = {'wave': config.getfloat('norm_wave'),
                      'wait': config.getfloat('norm_wait')}
        self.clips = {'wave': config.getfloat('clip_wave'),
                      'wait': config.getfloat('clip_wait')}
        self.coef_wait = config.getfloat('coef_wait')
        self.train_mode = True
        test_seeds = config.get('test_seeds').split(',')
        test_seeds = [int(s) for s in test_seeds]
        self._init_map() #??
        self.init_data(is_record, record_stats, output_path) #??
        self.init_test_seeds(test_seeds) #??
        self._init_sim(self.seed) #??
        self._init_nodes() #??
        self.terminate() #??
# defining parameters from the config file through init function

    def _debug_traffic_step(self):
        for node_name in self.node_names:  # iterating over the node_names
            node = self.nodes[node_name]  # finding node_name in nodes array and storing in node
            phase = self.sim.trafficlight.getRedYellowGreenState(self.node_names[0])  # where is it coming from? node_names[0] is the tls_id corresponding to which we get the phases
            cur_traffic = {'episode': self.cur_episode,
                           'time_sec': self.cur_sec,
                           'node': node_name,
                           'action': node.prev_action,
                           'phase': phase}  # storing node specific info in dictionary
            for i, ild in enumerate(node.ilds_in):
                cur_name = 'lane%d_' % i
                cur_traffic[cur_name + 'queue'] = self.sim.lane.getLastStepHaltingNumber(ild)  # Returns the total number of halting vehicles for the last time step on the given edge. A speed of less than 0.1 m/s is considered a halt.
                cur_traffic[cur_name + 'flow'] = self.sim.lane.getLastStepVehicleNumber(ild)  # The number of vehicles on this edge within the last time step.
                # cur_traffic[cur_name + 'wait'] = node.waits[i]
            self.traffic_data.append(cur_traffic)  # appending data to traffic_data array
# storing current traffic related info in curr_traffic dict with
# state variables like number of vehicles waiting (wait) and moving (wave)

    def _get_node_phase(self, action, node_name, phase_type):
        node = self.nodes[node_name]  # node name
        cur_phase = self.phase_map.get_phase(node.phase_id, action)  # getting phase id and action
        if phase_type == 'green':  # what's phase type
            return cur_phase
        prev_action = node.prev_action  # placing prev_action (what's this?)
        node.prev_action = action  # placing action
        if (prev_action < 0) or (action == prev_action):  # ?
            return cur_phase
        prev_phase = self.phase_map.get_phase(node.phase_id, prev_action) # phase corresponding to previous action
        switch_reds = []  # declaring array switch_reds
        switch_greens = []  # declaring array switch_greens
        for i, (p0, p1) in enumerate(zip(prev_phase, cur_phase)):
            if (p0 in 'Gg') and (p1 == 'r'):  # checking for prev_phase green and current phase red ( s/w to reds )
                switch_reds.append(i)
            elif (p0 in 'r') and (p1 in 'Gg'):  # checking for prev_phase red and current phase green ( s/w to greens )
                switch_greens.append(i)
        if not len(switch_reds):  # if there's no switching to red then we return cur_phase
            return cur_phase
        yellow_phase = list(cur_phase)  # creating yellow phase and assigning cur_phase list to it
        for i in switch_reds:
            yellow_phase[i] = 'y'  # changing all the previously switched to red lanes to y and storing in yellow_phase array
        for i in switch_greens:
            yellow_phase[i] = 'r'  # changing all the previously switched to green lanes to r and storing in yellow_phase array
        return ''.join(yellow_phase)   # returning elements of yellow phase =[a,b,c,d] as a b c d
# changing phases according to already existing phases

    def _get_node_phase_id(self, node_name):
        # needs to be overwritten
        raise NotImplementedError()

    def _get_node_state_num(self, node):
        assert len(node.lanes_in) == self.phase_map.get_lane_num(node.phase_id)  # ??
        # wait / wave states for each lane
        return len(node.ilds_in)

    def _get_state(self):
        # hard code the state ordering as wave, wait, fp
        state = []  # declaring an empty array called state
        # measure the most recent state
        self._measure_state_step()  # finding the recent state

        # get the appropriate state vectors
        for node_name in self.node_names:
            node = self.nodes[node_name]
            # wave is required in state
            if self.agent == 'greedy':  # checking if agent mentions greedy
                state.append(node.wave_state)  # appending the wave_state
            elif self.agent == 'a2c':  # checking if the agent is a2c
                if 'wait' in self.state_names:  # checking if wait is part of state_names
                    state.append(np.concatenate([node.wave_state, node.wait_state]))  # we append to state the concatenated wave_state and wait_state
                else:
                    state.append(node.wave_state)  # i think wait_state is optional while wave_state is mandatory
            else:
                cur_state = [node.wave_state]  #if agent is not greedy we have just wave_state as cur_state
                # include wave states of neighbors
                for nnode_name in node.neighbor:  # iterating through the neighbor
                    if self.agent != 'ma2c':  # for agent where its not ma2c
                        cur_state.append(self.nodes[nnode_name].wave_state)  # we just take the wave_states of all the neighboring nodes
                    else:
                        # discount the neigboring states
                        cur_state.append(self.nodes[nnode_name].wave_state * self.coop_gamma)  # in ma2c we discount the neighboring states ( spatial discount factor )
                # include wait state
                if 'wait' in self.state_names:  # checking if wait is in state_names
                    cur_state.append(node.wait_state)  # appending to cur_state
                # include fingerprints of neighbors
                if self.agent == 'ma2c':  # checking agent to be ma2c
                    for nnode_name in node.neighbor:  # iterating through neighbor
                        cur_state.append(self.nodes[nnode_name].fingerprint)  # fingerprint? also included in cur_state
                state.append(np.concatenate(cur_state))  # state appended with cur_state

        if self.agent == 'a2c':
            state = np.concatenate(state)

        # # clean up the state and fingerprint measurements
        # for node in self.node_names:
        #     self.nodes[node].state = np.zeros(self.nodes[node].num_state)
        #     self.nodes[node].fingerprint = np.zeros(self.nodes[node].num_fingerprint)
        return state

    def _init_nodes(self):
        nodes = {}  # declaring empty dict
        for node_name in self.sim.trafficlight.getIDList():  # iterating through ids of all the traffic lights in the scenario
            if node_name in self.neighbor_map:  # if the id of a particular traffic light is in neighbor map
                neighbor = self.neighbor_map[node_name]  # storing corr neighbor_map attribute in neighbor
            else:
                logging.info('node %s can not be found!' % node_name) # if node not in neighbor logged
                neighbor = []  # declaring an empty array
            nodes[node_name] = Node(node_name,
                                    neighbor=neighbor,
                                    control=True) # creating object for Node class
            # controlled lanes: l:j,i_k
            lanes_in = self.sim.trafficlight.getControlledLanes(node_name)  # storing list of lanes controlled by given traffic light(node_name)
            nodes[node_name].lanes_in = lanes_in  # storing above info in lanes_in attribute of nodes obj
            # controlled edges: e:j,i
            # lane ilds: ild:j,i_k for road ji, lane k.
            # edges_in = []
            ilds_in = []  # declaring an empty array
            for lane_name in lanes_in:  # iterating thru the lanes_in list
                ild_name = lane_name  # make ild_name as lane_name
                if ild_name not in ilds_in:
                    ilds_in.append(ild_name)  # appending lane_name values to ilds_in array
            # nodes[node_name].edges_in = edges_in
            nodes[node_name].ilds_in = ilds_in  # storing info in ilds_in attribute of nodes class
        self.nodes = nodes # ??
        self.node_names = sorted(list(nodes.keys())) # ??
        s = 'Env: init %d node information:\n' % len(self.node_names)  # message
        for node in self.nodes.values():  # iterating through values attribute of nodes obj
            s += node.name + ':\n'
            s += '\tneigbor: %r\n' % node.neighbor
            # s += '\tlanes_in: %r\n' % node.lanes_in  # creating string with all info relating to the node like the node name, neigbor nodes and ilds_in etc.
            s += '\tilds_in: %r\n' % node.ilds_in
            # s += '\tedges_in: %r\n' % node.edges_in
        logging.info(s)  # logging the above info s
        self._init_action_space()  #calling action_space init function
        self._init_state_space()  # calling state_space init function

    def _init_action_space(self):
        # for local and neighbor coop level
        self.n_a_ls = []  # declaring empty array
        for node_name in self.node_names:  # iterating thru node_names
            node = self.nodes[node_name]  # storing info from nodes class corr to node_name in node
            phase_id = self._get_node_phase_id(node_name)  # get phase_id not implemented
            node.phase_id = phase_id
            node.n_a = self.phase_map.get_phase_num(phase_id)  # get phase_num ( what's phase_num?? )
            self.n_a_ls.append(node.n_a)  # appending phase_num to n_a_ls array
        # for global coop level
        self.n_a = np.prod(np.array(self.n_a_ls))  # storing product of elements in n_a_ls in n_a

    def _init_map(self):
        # needs to be overwriteen
        self.neighbor_map = None
        self.phase_map = None
        self.state_names = None
        raise NotImplementedError()

    def _init_policy(self):
        policy = []  # declaring empty array policy
        for node_name in self.node_names:  # iterating thru node_names
            phase_num = self.nodes[node_name].n_a  # taking phase_num as n_a found by np.prod(self.n_a_ls)
            p = 1. / phase_num  # making p array equal to element wise 1/
            policy.append(np.array([p] * phase_num))  # appending to polciy array (p * phase num) but why??
        return policy

    def _init_sim(self, seed, gui=False):  # init simulation
        sumocfg_file = self._init_sim_config(seed)
        if gui:
            app = 'sumo-gui'
        else:
            app = 'sumo'
        command = [checkBinary(app), '-c', sumocfg_file]
        command += ['--seed', str(seed)]
        command += ['--remote-port', str(self.port)]
        command += ['--no-step-log', 'True']
        if self.name != 'real_net':
            command += ['--time-to-teleport', '600']  # long teleport for safety
        else:
            command += ['--time-to-teleport', '300']
        command += ['--no-warnings', 'True']
        command += ['--duration-log.disable', 'True']
        # collect trip info if necessary
        if self.is_record:
            command += ['--tripinfo-output',
                        self.output_path + ('%s_%s_trip.xml' % (self.name, self.agent))]
        subprocess.Popen(command)
        # wait 2s to establish the traci server
        time.sleep(2)
        self.sim = traci.connect(port=self.port)

    def _init_sim_config(self):
        # needs to be overwriteen
        raise NotImplementedError()

    def _init_sim_traffic(self):
        return

    def _init_state_space(self):
        self._reset_state()  # calling fn to reset_state
        self.n_s_ls = []  # declaring empty arrays
        self.n_w_ls = []
        self.n_f_ls = []
        for node_name in self.node_names:  # iterating thru all the node_names
            node = self.nodes[node_name]
            num_wave = node.num_state  # keeping num_state as num_wave
            num_fingerprint = 0  # keeping fingerprint to 0
            for nnode_name in node.neighbor:  # iterating thru neighbor nodes
                if self.agent not in ['a2c', 'greedy']:
                    # all marl agents have neighborhood communication
                    num_wave += self.nodes[nnode_name].num_state  # if agent is not a2c or greedy all marl agents have comm so we make num_wave added with neigboring nodes num_state too
                if self.agent == 'ma2c':  # checking for ma2c
                    # only ma2c uses neighbor's policy
                    num_fingerprint += self.nodes[nnode_name].num_fingerprint  # as it uses neighbor's policy we update fingerprint to that of neighbor nodes too
            num_wait = 0 if 'wait' not in self.state_names else node.num_state  # if wait is not present in state_names we give wait = 0 else we give it node.num_state
            self.n_s_ls.append(num_wave + num_wait + num_fingerprint)  # updating above declared arrays with total num_wave + num_wait + num_fingerprint for each node in n_s_ls
            self.n_f_ls.append(num_fingerprint)  # updating n_f_ls with fingerprint for each node
            self.n_w_ls.append(num_wait)  # updating n_w_ls with wait time of each node
        self.n_s = np.sum(np.array(self.n_s_ls))  # n_s is updated as sum of all elements in n_s_ls

    def _measure_reward_step(self):
        rewards = []  # declaring empty array called reward
        for node_name in self.node_names:  # iterating thru all node_names
            queues = []  # declaring empty array called queues
            waits = []  # declaring empty array called waits
            for ild in self.nodes[node_name].ilds_in:  # iterating thu ilds_in
                if self.obj in ['queue', 'hybrid']:   #checking if objective is just que or hybrid in the config file
                    if self.name == 'real_net':  # checking if real_net
                        cur_queue = min(10, self.sim.lane.getLastStepHaltingNumber(ild))  # making cur_que as the total number of halting vehicles in particular lane at last time step
                    else:
                        cur_queue = self.sim.lanearea.getLastStepHaltingNumber(ild)  # for non real_net n/w we use lanearea detectors and do the same
                    queues.append(cur_queue)  # then we append to cur_queue
                if self.obj in ['wait', 'hybrid']:  # checking if objective is in wait, hybrid
                    max_pos = 0  # declaring max_pos = 0
                    car_wait = 0  # declaring car_wait = 0
                    if self.name == 'real_net':
                        cur_cars = self.sim.lane.getLastStepVehicleIDs(ild)  # storing list of all vehicles in that lane in the last time step in cur_cars
                    else:
                        cur_cars = self.sim.lanearea.getLastStepVehicleIDs(ild)
                    for vid in cur_cars:  # iterating thru vehicle ids in cur_cars
                        car_pos = self.sim.vehicle.getLanePosition(vid)  #position of a vehicle corr to its veh_id
                        if car_pos > max_pos:
                            max_pos = car_pos
                            car_wait = self.sim.vehicle.getWaitingTime(vid)  # finding waiting time of farthest vehicle
                    waits.append(car_wait)  # appending to waits
                # if self.name == 'real_net':
                #     lane_name = ild.split(':')[1]
                # else:
                #     lane_name = 'e:' + ild.split(':')[1]
                # queues.append(self.sim.lane.getLastStepHaltingNumber(lane_name))

            queue = np.sum(np.array(queues)) if len(queues) else 0  # making queue as sum of all entries
            wait = np.sum(np.array(waits)) if len(waits) else 0  # making wait as sum of all its entries
            # if self.obj in ['wait', 'hybrid']:
            #     wait = np.sum(self.nodes[node_name].waits * (queues > 0))
            if self.obj == 'queue':  # if objective just has queue we have
                reward = - queue # reward as -q
            elif self.obj == 'wait':  # if objective is only wait we have
                reward = - wait  # reward as -wait
            else:  # otherwise we have reward as -q - self.coef_wait (weighting parameter wrt q for wait) * wait
                reward = - queue - self.coef_wait * wait
            rewards.append(reward) # appending to rewards
        return np.array(rewards)  # making it an np.array

    def _measure_state_step(self):
        for node_name in self.node_names:
            node = self.nodes[node_name]
            for state_name in self.state_names:  # iterating over state_names
                if state_name == 'wave':  #checking if state_name is wave
                    cur_state = []  # declaring empty array cur_state
                    for ild in node.ilds_in:  # what is ilds_in??
                        if self.name == 'real_net':  # checking if name is real_net
                            cur_wave = self.sim.lane.getLastStepVehicleNumber(ild)  # in that case we get the wave (flow) corresponding to the lane
                        else:
                            cur_wave = self.sim.lanearea.getLastStepVehicleNumber(ild)  # if the name isn't real_wave we get the flow in the corr lane area detector (for a particular length)
                        cur_state.append(cur_wave)  # append to cur_state array declared above
                    cur_state = np.array(cur_state)  # make cur_state a numpy array
                else:  # for state that's not wave, wait (here)
                    cur_state = []  # declaring empty array
                    for ild in node.ilds_in:
                        max_pos = 0
                        car_wait = 0
                        if self.name == 'real_net':  # checking if name us real_net
                            cur_cars = self.sim.lane.getLastStepVehicleIDs(ild)  # returns the list of vehicle ids of all cars on that lane for the last time step
                        else:
                            cur_cars = self.sim.lanearea.getLastStepVehicleIDs(ild)  # same as above except that we use a lane area detector here ( why? )
                        for vid in cur_cars:  # iterating through the vehicle ids in cur_cars
                            car_pos = self.sim.vehicle.getLanePosition(vid)  # distance of the front bumper of the vehicle from the lane starting
                            if car_pos > max_pos:  # checking if distance of car is greater than the max_pos
                                max_pos = car_pos  #then changing max_pos to car_pos
                                car_wait = self.sim.vehicle.getWaitingTime(vid)  # find the waiting time of all  the vehicle (time spent in speed < 0.1 m/s)
                        cur_state.append(car_wait) # appending the cur_state array with waiting time of the vehicles
                    cur_state = np.array(cur_state)  # making it a numpy array
                if self.record_stats:  # ??
                    self.state_stat[state_name] += list(cur_state)  # ??
                # normalization
                norm_cur_state = self._norm_clip_state(cur_state,
                                                       self.norms[state_name],   # calling function _norm_clip_state and we pass the cur_state array from before to normalise
                                                       self.clips[state_name])  # with norms for each state_name stored in norms[state_name] array and clip values of each state_name is stored in clips[state_name]
                if state_name == 'wave':
                    node.wave_state = norm_cur_state  # based on state_name we store the normalised node.wave_state or node.wait_state
                else:
                    node.wait_state = norm_cur_state
# function to get state variable values from sumo environment and normalise them
    def _measure_traffic_step(self):
        cars = self.sim.vehicle.getIDList()  # ids of all vehicles running in the scenario
        num_tot_car = len(cars)  # total number of cars as len(cars array above)
        num_in_car = self.sim.simulation.getDepartedNumber()  # number of incoming cars
        num_out_car = self.sim.simulation.getArrivedNumber()  # number of outgoing cars
        if num_tot_car > 0:
            avg_waiting_time = np.mean([self.sim.vehicle.getWaitingTime(car) for car in cars]) # finding average waiting time for all cars
            avg_speed = np.mean([self.sim.vehicle.getSpeed(car) for car in cars]) # finding avg_speed for all cars
        else:
            avg_speed = 0  # if no cars are in scenario then making avg_waiting time and avg_speed = 0
            avg_waiting_time = 0
        # all trip-related measurements are not supported by traci,
        # need to read from outputfile afterwards
        queues = []  # declaring empty array queues
        for node_name in self.node_names:  # iterating thru node_names
            for ild in self.nodes[node_name].ilds_in:  # iterating thu lanes
                queues.append(self.sim.lane.getLastStepHaltingNumber(ild))  # finding number of vehicles halted in last time step
        avg_queue = np.mean(np.array(queues))  # avg_queue as average of all queues across all lanes under a node
        std_queue = np.std(np.array(queues))  # std_queues as std of all queues across all lanes under a node
        cur_traffic = {'episode': self.cur_episode,
                       'time_sec': self.cur_sec,
                       'number_total_car': num_tot_car,
                       'number_departed_car': num_in_car,
                       'number_arrived_car': num_out_car,
                       'avg_wait_sec': avg_waiting_time,
                       'avg_speed_mps': avg_speed,
                       'std_queue': std_queue,
                       'avg_queue': avg_queue}  # updating cur_traffic dict with relevant info
        self.traffic_data.append(cur_traffic)  # append dict to traffic_data

    @staticmethod
    def _norm_clip_state(x, norm, clip=-1):
        x = x / norm  # normalising wrt norm
        return x if clip < 0 else np.clip(x, 0, clip)  # if clip < 0 we return x or else we return clip(x,o,clip) where x < 0 gives x = 0 and x > clip gives x = clip

    def _reset_state(self):
        for node_name in self.node_names:  # iterating thru node_names
            node = self.nodes[node_name]  # storing nodes obj node_name's attributes in node
            # prev action for yellow phase before each switch
            node.prev_action = 0  # making prev_action = 0
            # fingerprint is previous policy[:-1]
            node.num_fingerprint = node.n_a - 1  # making fingerprint as node.n_a-1 ??
            node.num_state = self._get_node_state_num(node)  # num_state for node
            # node.waves = np.zeros(node.num_state)
            # node.waits = np.zeros(node.num_state)

    def _set_phase(self, action, phase_type, phase_duration):
        for node_name, a in zip(self.node_names, list(action)):  # iterating thru node_names and list of actions simultaneously
            phase = self._get_node_phase(a, node_name, phase_type)  # getting phase_info
            self.sim.trafficlight.setRedYellowGreenState(node_name, phase)  # setting the given phase until the next call
            self.sim.trafficlight.setPhaseDuration(node_name, phase_duration)  # setting the phase duration

    def _simulate(self, num_step):
        # reward = np.zeros(len(self.control_node_names))
        for _ in range(num_step): # iterating thru the num_steps
            self.sim.simulationStep()  # sumo simulation step advancement
            # self._measure_state_step()
            # reward += self._measure_reward_step()
            self.cur_sec += 1  # cur_Sec incremented
            if self.is_record:  # checking for is_record to be true
                # self._debug_traffic_step()
                self._measure_traffic_step()  # if is_record is true we call _measure_traffic_step
        # return reward

    def _transfer_action(self, action):
        '''  Transfer global action to a list of local actions  '''
        phase_nums = []  # declaring phase_nums empty array
        for node in self.control_node_names:  # iterating thru control_node_names ?
            phase_nums.append(self.nodes[node].phase_num)  # appending the num_phase ?
        action_ls = []  # declaring action_ls array
        for i in range(len(phase_nums) - 1):  # iterating thru 1 to phase_nums length
            action, cur_action = divmod(action, phase_nums[i])  # action = action/phase_nums[i] cur_action = action % phase_nums[i]
            action_ls.append(cur_action)  # action_ls appended with cur_action
        action_ls.append(action)  # action_ls appended with action
        return action_ls

    def _update_waits(self, action):
        for node_name, a in zip(self.node_names, action):  # simultaenously iterating node_names and actions
            red_lanes = set()  # declaring red_lanes as set variable
            node = self.nodes[node_name]  # node as node_name
            for i in self.phase_map.get_red_lanes(node.phase_id, a):  # iterating thru get_red lanes return
                red_lanes.add(node.lanes_in[i])  # appending those from get_red_lanes in red_lanes set
            for i in range(len(node.waits)):  # iterating thru len(waits)
                lane = node.ilds_in[i]  # storing lane_id in lane
                if lane in red_lanes:
                    node.waits[i] += self.control_interval_sec  # if the lane is in the red_lanes adding waiting time
                else:
                    node.waits[i] = 0  # else no waiting time

    def collect_tripinfo(self):
        # read trip xml, has to be called externally to get complete file
        trip_file = self.output_path + ('%s_%s_trip.xml' % (self.name, self.agent))  # name_agent_trip.xml in output_path
        tree = ET.ElementTree(file=trip_file)  # element tree class object
        for child in tree.getroot():
            cur_trip = child.attrib  # ??
            cur_dict = {}
            cur_dict['episode'] = self.cur_episode
            cur_dict['id'] = cur_trip['id']
            cur_dict['depart_sec'] = cur_trip['depart']
            cur_dict['arrival_sec'] = cur_trip['arrival']
            cur_dict['duration_sec'] = cur_trip['duration']
            cur_dict['wait_step'] = cur_trip['waitingCount']
            cur_dict['wait_sec'] = cur_trip['waitingTime']
            self.trip_data.append(cur_dict)
        # delete the current xml  why ??
        cmd = 'rm ' + trip_file
        subprocess.check_call(cmd, shell=True)

    def init_data(self, is_record, record_stats, output_path):
        self.is_record = is_record
        self.record_stats = record_stats
        self.output_path = output_path
        if self.is_record:
            self.traffic_data = []
            self.control_data = []
            self.trip_data = []
        if self.record_stats:
            self.state_stat = {}
            for state_name in self.state_names:
                self.state_stat[state_name] = []

    def init_test_seeds(self, test_seeds):
        self.test_num = len(test_seeds)
        self.test_seeds = test_seeds

    def output_data(self):
        if not self.is_record:
            logging.error('Env: no record to output!')
        control_data = pd.DataFrame(self.control_data)
        control_data.to_csv(self.output_path + ('%s_%s_control.csv' % (self.name, self.agent)))
        traffic_data = pd.DataFrame(self.traffic_data)
        traffic_data.to_csv(self.output_path + ('%s_%s_traffic.csv' % (self.name, self.agent)))
        trip_data = pd.DataFrame(self.trip_data)
        trip_data.to_csv(self.output_path + ('%s_%s_trip.csv' % (self.name, self.agent)))

    def reset(self, gui=False, test_ind=0):
        # have to terminate previous sim before calling reset
        self._reset_state()
        if self.train_mode:
            seed = self.seed
        else:
            seed = self.test_seeds[test_ind]
        # self._init_sim(gui=True)
        self._init_sim(seed, gui=gui)
        self.cur_sec = 0
        self.cur_episode += 1
        # initialize fingerprint
        if self.agent == 'ma2c':
            self.update_fingerprint(self._init_policy())
        self._init_sim_traffic()
        # next environment random condition should be different
        self.seed += 1
        return self._get_state()

    def terminate(self):
        self.sim.close()

    def step(self, action):
        if self.agent == 'a2c':
            action = self._transfer_action(action)
        # self._update_waits(action)
        self._set_phase(action, 'yellow', self.yellow_interval_sec)
        self._simulate(self.yellow_interval_sec)
        rest_interval_sec = self.control_interval_sec - self.yellow_interval_sec
        self._set_phase(action, 'green', rest_interval_sec)
        self._simulate(rest_interval_sec)
        state = self._get_state()
        reward = self._measure_reward_step()
        done = False
        if self.cur_sec >= self.episode_length_sec:
            done = True
        global_reward = np.sum(reward)  # for fair comparison
        if self.is_record:
            action_r = ','.join(['%d' % a for a in action])
            cur_control = {'episode': self.cur_episode,
                           'time_sec': self.cur_sec,
                           'step': self.cur_sec / self.control_interval_sec,
                           'action': action_r,
                           'reward': global_reward}
            self.control_data.append(cur_control)

        # use local rewards in test
        if not self.train_mode:
            return state, reward, done, global_reward
        if self.agent in ['a2c', 'greedy']:
            reward = global_reward
        elif self.agent != 'ma2c':
            # global reward is shared in independent rl
            new_reward = [global_reward] * len(reward)
            reward = np.array(new_reward)
            if self.name == 'real_net':
                # reward normalization in env for realnet
                reward = reward / (len(self.node_names) * REALNET_REWARD_NORM)
        else:
            # discounted global reward for ma2c
            new_reward = []
            for node_name, r in zip(self.node_names, reward):
                cur_reward = r
                for nnode_name in self.nodes[node_name].neighbor:
                    i = self.node_names.index(nnode_name)
                    cur_reward += self.coop_gamma * reward[i]
                # for i, nnode in enumerate(self.node_names):
                #     if nnode == node:
                #         continue
                #     if nnode in self.nodes[node].neighbor:
                #         cur_reward += self.coop_gamma * reward[i]
                #     elif self.name == 'small_grid':
                #         # in small grid, agent is at most 2 steps away
                #         cur_reward += (self.coop_gamma ** 2) * reward[i]
                #     else:
                #         # in large grid, a distance map is used
                #         if nnode in self.distance_map[node]:
                #             distance = self.distance_map[node][nnode]
                #             cur_reward += (self.coop_gamma ** distance) * reward[i]
                #         else:
                #             cur_reward += (self.coop_gamma ** self.max_distance) * reward[i]
                if self.name != 'real_net':
                    new_reward.append(cur_reward)
                else:
                    n_node = 1 + len(self.nodes[node_name].neighbor)
                    new_reward.append(cur_reward / (n_node * REALNET_REWARD_NORM))
            reward = np.array(new_reward)
        return state, reward, done, global_reward

    def update_fingerprint(self, policy):
        for node_name, pi in zip(self.node_names, policy):
            self.nodes[node_name].fingerprint = np.array(pi)[:-1]
