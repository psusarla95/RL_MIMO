'''
Author: psusarla
Date: 10.10.2018

Problem: Two BS with one antenna unit on each side
Parameter considered: Power transmitter levels
Solution: Q-learning
States - {28,29,30,31,32,33,34,35,36,37,38} SNR dBm levels
delta - 14
actions - {-2,-1,0,1,2} Power transmitter levels
rewards - 1 if SINR(i) > delta; -1 otherwise
control policy - efficient beamforming by optimizing parameters
'''

import numpy as np
from scipy.constants import *
import cmath
import math


#conversion from dB to linear
def db2lin(val_db):
    val_lin = 0
    #for i in range(1, val_db):
    val_lin = 10**(val_db/10)
    return val_lin

#cosine over degree
def cosd(val):
    return math.cos(val*pi/180)

#sine over degree
def sind(val):
    return math.sin(val*pi/180)

def max_dict(d):
    max_key = None
    max_val = float('-inf')

    for k, v in d.items():
        if v > max_val:
            max_key = k
            max_val = v
    return max_key, max_val

def random_action(a, eps=0.1):
    p = np.random.random()
    #epsilon-soft method (also known as epsilon-greedy from now on)
    if p < 1-eps:
        return a
    else:
        return np.random.choice(ALL_POSSIBLE_ACTIONS)

#print_values(beam)

class MIMO:
    def __init__(self):
        self.freq = 28e9  # 28 GHz
        self.d = 0.5 #relative element space
        self.l = c/self.freq  #c - speed of light, scipy constant

        #transmitter and receiver location
        self.X_range = 108
        self.X_angle = 0
        x = self.X_range*cosd(self.X_angle)
        y = self.X_range*sind(self.X_angle)
        X = [x,y] #row list of x,y

        self.X_t = X[0]
        self.X_r = X[1]
        self.P_tx = 30  # dBm

    def Transmit_Energy(self, beta):
        df = 75e3 #carrier spacing frequency
        nFFT = 2048 #no. of subspace carriers

        T_sym = 1/df
        B = nFFT * df



        Es = db2lin(self.P_tx + beta)*(10**(-3)/B)
        self.P_tx += beta
        return Es

    def Channel(self):
        self.Dist = np.linalg.norm(np.array(self.X_t)- np.array(self.X_r))
        FSL = 20*np.log10(self.Dist) + 20*np.log10(self.freq) - 147.55 #db, free space path loss
        channel_loss = db2lin(-FSL)
        g_c = np.sqrt(channel_loss)
        h = g_c*cmath.exp(-1j*(pi/4)) #LOS channel coefficient
        return h

    def array_factor(self, ang, n):
        x = np.arange(0,n)
        y = np.array([1 / np.sqrt(n) * np.exp(1j * 2 * pi * self.d / self.l*math.sin(ang)*k) for k in x])
        #print("y: {0}".format(y))
        return y

    def Antenna_Array(self):
        self.N_tx = 4 #no. of transmitting antennas
        self.N_rx = 4 #no. of receiving antennas

        if self.X_r > 0:
            theta_tx = math.acos(self.X_t/self.Dist)
        else:
            theta_tx = -1*math.acos(self.X_t/self.Dist)

        alpha = 0 #relative rotation between transmit and receiver arrays

        phi_rx = theta_tx - pi + alpha
        #phi_rx = pi + theta_tx - alpha
        #phi_rx = pi

        a_tx = self.array_factor(theta_tx, self.N_tx)
        a_rx = self.array_factor(phi_rx, self.N_rx)

        #print("Theta_tx: {0}".format(theta_tx))
        #print("Phi_RX: {0}".format(phi_rx))
        return a_tx, a_rx, self.N_tx, self.N_rx

    def Noise(self):
        N0dBm = -174
        N0 = db2lin(N0dBm)*(10**-3)
        return N0

    def Calc_SNR(self, beta):
        Es = self.Transmit_Energy(beta)
        h = self.Channel()
        a_tx, a_rx, N_tx, N_rx = self.Antenna_Array()
        N0 = self.Noise()

        val = h*np.sqrt(N_tx)*a_tx.dot(a_tx)*a_rx.dot(a_rx)*np.sqrt(N_rx)
        SNR = Es * np.linalg.norm(val)**2 / N0

        #print("atx:{0}".format(a_tx))
        #print("arx:{0}".format(a_rx))
        #print("action: {0}".format(beta))
        #print("h: {0}".format(h))
        #print("Es: {0}".format(Es))
        #print("val:{0} ".format(val))
        #print("N0: {0}".format(N0))
        #print("X_t: {0}".format(self.X_t))
        #print("X_r: {0}".format(self.X_r))

        return SNR

class Beam:
    def __init__(self, states, mimo, start):
        self.min_state = np.min(states)
        self.max_state = np.max(states)
        self.current_state = start
        self.mimo = mimo

    def set(self, rewards, actions):
        # rewards is a dict of (i,j) : r (of that state) reward
        # actions is a dict of (i,j): actions(from that state) => list of possible actions

        self.rewards = rewards
        self.actions = actions

    def set_state(self, s):
        self.current_state = s

    def get_current_state(self):
        return self.current_state

    def is_terminal(self, s):
        if s not in self.actions:
            return True
        else:
            return False

    def move(self, action):
        #check the legal move first and then return its reward
        if action in self.actions[self.current_state]:
            SNR = self.mimo.Calc_SNR(action)
            # print("Calculated SNR:{0}".format(SNR))
            if self.current_state > self.max_state:
                self.current_state = self.max_state
            elif self.current_state < self.min_state:
                self.current_state = self.min_state
            else:
                self.current_state = int(np.round(10*np.log10(SNR)))

        return self.rewards.get((self.current_state), 0)

    '''
    def undo_move(self,action):
        self.current_state += action
    '''

    def game_over(self):
        #print("Game over: {0}".format(self.current_state))
        if (self.max_state == self.current_state) or (self.current_state not in self.actions):
            return True
        else:
            return False

    def all_states(self):
        return set(self.actions.keys()) | set(self.rewards.keys())


ALL_POSSIBLE_STATES=[28,29,30,31,32,33,34,35,36,37,38] #SNR dBm levels
ALL_POSSIBLE_ACTIONS=[-2,-1, 0, 1, 2]
start_state = 29
DELTA = 32
GAMMA = 0.9
ALPHA = 0.1

if __name__ == '__main__':

    mimo = MIMO()
    beam = Beam(ALL_POSSIBLE_STATES, mimo, start_state)

    #state and its corresponding options
    actions ={
        28 : {0, 1, 2},
        29 : {-1, 0, 1, 2},
        30 : {-2, -1, 0, 1, 2},
        31 : {-2, -1, 0, 1, 2},
        32 : {-2, -1, 0, 1, 2},
        33 : {-2, -1, 0, 1, 2},
        34 : {-2, -1, 0, 1, 2},
        35 : {-2, -1, 0, 1, 2},
        36 : {-2, -1, 0, 1, 2},
        37 : {-2, -1, 0, 1},
        38 : {-2, -1, 0}
    }

    #rewards is also a dictionary

    rewards = dict.fromkeys(actions.keys())
    for key in rewards.keys():
        if key >= DELTA:
            rewards[key] = 1
        else:
            rewards[key] = -1
    print("Rewards: {0}".format(rewards))
    beam.set(rewards, actions)

    # no policy initialization, we will derive our policy from most recent Q
    #initialize Q(s,a)
    Q = {}
    states = beam.all_states()
    for s in states:
        Q[s] = {}
        for a in ALL_POSSIBLE_ACTIONS:
            Q[s][a] = 0


    update_counts ={} #to keep track of how many times each state gonna get updated
    update_counts_sa={} #for adaptive learning rate
    for s in states:
        update_counts_sa[s] = {}
        for a in ALL_POSSIBLE_ACTIONS:
            update_counts_sa[s][a] = 1.0

    #repeat until converrgence
    t = 1.0
    deltas = []
    for iter in range(1000):
        if iter % 50 == 0:
            t += 10e-3
        if iter % 100 == 0:
            print("%d iterations are done" % iter)

        s = start_state
        beam.set_state(s)
        a = (max_dict(Q[s]))[0]


        biggest_change = 0
        while not beam.game_over():
            a = random_action(a, eps=0.5 / t)
            r = beam.move(a)
            s2 = beam.get_current_state()
            if (s2 > np.max(ALL_POSSIBLE_STATES)):
                break

            alpha = ALPHA / update_counts_sa[s][a]
            update_counts_sa[s][a] += 0.005

            old_qsa = Q[s][a]
            a2, max_q_s2a2 = max_dict(Q[s2])
            Q[s][a] = Q[s][a] + alpha*(r + GAMMA*max_q_s2a2 - Q[s][a])
            biggest_change = max(biggest_change, np.abs(old_qsa - Q[s][a]))

            update_counts[s] = update_counts.get(s,0) + 1

            s = s2
            a = a2

        deltas.append(biggest_change)

        print("Q: ")
        for s in ALL_POSSIBLE_STATES:
            for a in ALL_POSSIBLE_ACTIONS:
                print("s: {0}, a: {1}, Q: {2}".format(s,a, Q[s][a]))

    #plt.plot(deltas)
    #plt.show()
