'''
Author: psusarla
Date: 15.10.2018

Problem: Two BS with one antenna unit on each side
Parameter considered: Power transmitter levels, Angle of arrival and Angle of departure
Solution: Q-learning
States - {28,29,30,31,32,33,34,35,36,37,38} SNR dBm levels
delta - 36
actions - {-2,-1,0,1,2} Power transmitter levels,{-pi/2, -pi/4, 0, pi/4, pi/2} angle of departure and angle of arrival levels
rewards - 1 if SINR(i) > delta; -1 otherwise
control policy - efficient beamforming by optimizing parameters
'''

import numpy as np
from scipy.constants import *
import cmath
import math
import matplotlib.pyplot as plt


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

def get_Item(lst, elem, err):
    try:
        return lst[elem]
    except IndexError:
        return err

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
        ndx = np.random.choice(len(ALL_POSSIBLE_ACTIONS))
        return ALL_POSSIBLE_ACTIONS[ndx]

#print_values(beam)
def distance(s_1,s_g):
    return np.linalg.norm(s_1-s_g)

class MIMO:
    def __init__(self, PwT_actions, AoA_actions):
        self.PwT_actions = PwT_actions
        self.AoA_actions = AoA_actions
        self.PwT_ndx = 0
        self.AoA_ndx = 0
        self.freq = 28e9  # 28 GHz
        self.d = 0.5 #relative element space
        self.l = c/self.freq  #c - speed of light, scipy constant

        #transmitter and receiver location
        self.X_range = 108
        self.X_angle = 0
        self.P_tx = 30  # dBm
        self.AoD = pi #0 radians, range [-pi/2, pi/2]
        self.AoA = 0 #0 raidans, range [-pi/2, pi/2]

        x = self.X_range*cosd(self.X_angle)
        y = self.X_range*sind(self.X_angle)
        X = [x,y] #row list of x,y

        self.X_t = X[0]
        self.X_r = X[1]
        self.Dist = np.linalg.norm(np.array(self.X_t)- np.array(self.X_r))


    def Transmit_Energy(self, beta_sym):
        df = 75e3 #carrier spacing frequency
        nFFT = 2048 #no. of subspace carriers

        T_sym = 1/df
        B = nFFT * df


        if beta_sym == '0':
            i = 0
        elif beta_sym == '-':
            i = -1
        else:
            i = 1
        self.PwT_ndx += i
        beta = get_Item(self.PwT_actions, self.PwT_ndx, -100) #-100 represents illegal PwT index, index out of range

        if beta == -100:
            self.PwT_ndx -= i
            return "error"
        else:
            Es = db2lin(self.P_tx + beta)*(10**(-3)/B)
            self.P_tx += beta
            return Es

    def Channel(self):

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

    def Antenna_Array(self, beta_sym):
        self.N_tx = 4 #no. of transmitting antennas
        self.N_rx = 4 #no. of receiving antennas


        if self.X_r > 0:
            self.theta_tx = math.acos(self.X_t/self.Dist)
        else:
            self.theta_tx = -1*math.acos(self.X_t/self.Dist)

        alpha = 0 #relative rotation between transmit and receiver arrays

        self.phi_rx = self.theta_tx - pi + alpha

        #phi_rx = pi + theta_tx - alpha
        #phi_rx = pi

        a_tx = self.array_factor(self.theta_tx, self.N_tx)
        a_rx = self.array_factor(self.phi_rx, self.N_rx)

        if beta_sym == '0':
            i = 0
        elif beta_sym == '-':
            i = -1
        else:
            i = 1
        self.AoA_ndx += i
        beta = get_Item(self.AoA_actions, self.AoA_ndx, -100) #-100 represents illegal PwT index, index out of range

        if beta == -100:
            self.AoA_ndx -= i
            return "error"
        else:
            w_vec = self.Communication_Vector(self.AoA_actions[self.AoA_ndx], self.N_tx) #transmit unit norm vector
            f_vec = self.Communication_Vector(self.AoD, self.N_rx) #receive unit norm vector

        #print("w_vec: {0}".format(w_vec))
        #print("f_vec: {0}".format(f_vec))
        #print("Theta_tx: {0}".format(theta_tx))
        #print("Phi_RX: {0}".format(phi_rx))
        return a_tx, a_rx, self.N_tx, self.N_rx, w_vec, f_vec

    def Noise(self):
        N0dBm = -174
        N0 = db2lin(N0dBm)*(10**-3)
        return N0

    #function to define tranmsit or receive unit norm vector
    def Communication_Vector(self, ang, n):
        x = np.arange(0,n)
        y = np.array([np.exp(1j * 2 * pi * 0.5 *math.sin(ang)*k) for k in x])
        return y

    def Calc_SNR(self, beta_pair):

        Es = self.Transmit_Energy(beta_pair[0])
        h = self.Channel()
        antenna_ret = self.Antenna_Array(beta_pair[1])
        if Es == 'error' or antenna_ret == 'error':
            return 'error'
        else:

            a_tx, a_rx, N_tx, N_rx, w_vec, f_vec = antenna_ret
            N0 = self.Noise()

            val = h*np.sqrt(N_rx)*w_vec.conj().T*a_rx*a_tx.conj().T*f_vec*np.sqrt(N_tx)
            #val = h*np.sqrt(N_tx)*a_tx.dot(a_tx)*a_rx.dot(a_rx)*np.sqrt(N_rx)
            SNR = Es * np.linalg.norm(val)**2 / N0
            #SNR = Es * np.absolute(val) ** 2 / N0

            #print("atx:{0}".format(a_tx))
            #print("arx:{0}".format(a_rx))
            #print("action: {0}".format(beta))
            #print("h: {0}".format(h))
            #print("Es: {0}".format(Es))
           # print("val:{0}, 2-norm of val: {1}".format(val, np.linalg.norm(val)**2))
            #print("Actual SNR: {0}".format(SNR))
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
            if SNR == 'error':
                return -1
            else:
                # print("Calculated SNR:{0}".format(SNR))
                '''
                The below logic is to ensure that we select the optimal values of parameters like power transmitter and AoD, only within the chosen SNR range
                '''
                logSNR = int(np.round(10*np.log10(SNR)))
                if logSNR > self.max_state:
                    self.current_state = self.max_state
                elif logSNR < self.min_state:
                    self.current_state = self.min_state
                else:
                    self.current_state = logSNR
                #print("Calculated SNR:{0}".format(self.current_state))


        r = self.rewards.get(self.current_state, 0)
        #if r == 0:
        #    print('a:{0}'.format(action))
        #print("reward: {0}".format(r))
        return r

        #return -1

    '''
    def undo_move(self,action):
        self.current_state += action
    '''

    def game_over(self, s):
        #print("Game over: {0}".format(self.current_state))
        if (s >= self.max_state) or (self.current_state not in self.actions.keys()):
            #print("Yes!! game is over!!")
            return True
        else:
            return False

    def all_states(self):
        return set(self.actions.keys()) | set(self.rewards.keys())

ALL_POSSIBLE_STATES=[21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38] #SNR dBm levels
ALL_POSSIBLE_PwT_ACTIONS=[-2,-1, 0, 1, 2]
ALL_POSSIBLE_AoA_ACTIONS=[-pi/2,-pi/4, 0, pi/4, pi/2]
ALL_POSSIBLE_ACTIONS=[('0', '0'), ('0', '-'), ('0', '+'), ('+', '0'), ('+', '-'), ('+', '+'), ('-', '0'), ('-', '-'), ('-', '+')]
start_state = 22
DELTA = 37
GAMMA = 0.9
ALPHA = 0.1
BETA=5
SIGMA=3.5
INITQ_DELTA=0.1
if __name__ == '__main__':

    mimo = MIMO(ALL_POSSIBLE_PwT_ACTIONS, ALL_POSSIBLE_AoA_ACTIONS)
    beam = Beam(ALL_POSSIBLE_STATES, mimo, start_state)

    #state and its corresponding options
    actions ={
        21: {('0', '0'), ('0', '-'), ('0', '+'), ('+', '0'), ('+', '-'), ('+', '+')},
        22: {('0', '0'), ('0', '-'), ('0', '+'), ('+', '0'), ('+', '-'), ('+', '+'), ('-', '0'), ('-', '-'),('-', '+')},
        23: {('0', '0'), ('0', '-'), ('0', '+'), ('+', '0'), ('+', '-'), ('+', '+'), ('-', '0'), ('-', '-'),('-', '+')},
        24: {('0', '0'), ('0', '-'), ('0', '+'), ('+', '0'), ('+', '-'), ('+', '+'), ('-', '0'), ('-', '-'),('-', '+')},
        25: {('0', '0'), ('0', '-'), ('0', '+'), ('+', '0'), ('+', '-'), ('+', '+'), ('-', '0'), ('-', '-'),('-', '+')},
        26: {('0', '0'), ('0', '-'), ('0', '+'), ('+', '0'), ('+', '-'), ('+', '+'), ('-', '0'), ('-', '-'),('-', '+')},
        27: {('0', '0'), ('0', '-'), ('0', '+'), ('+', '0'), ('+', '-'), ('+', '+'), ('-', '0'), ('-', '-'),('-', '+')},
        28: {('0', '0'), ('0', '-'), ('0', '+'), ('+', '0'), ('+', '-'), ('+', '+'), ('-', '0'), ('-', '-'),('-', '+')},
        29 : {('0', '0'), ('0', '-'), ('0', '+'), ('+', '0'), ('+', '-'), ('+', '+'), ('-', '0'), ('-', '-'), ('-', '+')},
        30 : {('0', '0'), ('0', '-'), ('0', '+'), ('+', '0'), ('+', '-'), ('+', '+'), ('-', '0'), ('-', '-'), ('-', '+')},
        31 : {('0', '0'), ('0', '-'), ('0', '+'), ('+', '0'), ('+', '-'), ('+', '+'), ('-', '0'), ('-', '-'), ('-', '+')},
        32 : {('0', '0'), ('0', '-'), ('0', '+'), ('+', '0'), ('+', '-'), ('+', '+'), ('-', '0'), ('-', '-'), ('-', '+')},
        33 : {('0', '0'), ('0', '-'), ('0', '+'), ('+', '0'), ('+', '-'), ('+', '+'), ('-', '0'), ('-', '-'), ('-', '+')},
        34 : {('0', '0'), ('0', '-'), ('0', '+'), ('+', '0'), ('+', '-'), ('+', '+'), ('-', '0'), ('-', '-'), ('-', '+')},
        35 : {('0', '0'), ('0', '-'), ('0', '+'), ('+', '0'), ('+', '-'), ('+', '+'), ('-', '0'), ('-', '-'), ('-', '+')},
        36 : {('0', '0'), ('0', '-'), ('0', '+'), ('+', '0'), ('+', '-'), ('+', '+'), ('-', '0'), ('-', '-'), ('-', '+')},
        37 : {('0', '0'), ('0', '-'), ('0', '+'), ('+', '0'), ('+', '-'), ('+', '+'), ('-', '0'), ('-', '-'), ('-', '+')},
        38 : {('0', '0'), ('0', '-'), ('0', '+'), ('-', '0'), ('-', '-'), ('-', '+')}
    }

    #rewards is also a dictionary

    rewards = dict.fromkeys(actions.keys())
    r_g = 1
    r_inf = -1
    s_g = np.max(ALL_POSSIBLE_STATES)
    for key in rewards.keys():
        '''
        if key >= DELTA:
            rewards[key] = r_g
        else:
            rewards[key] = r_inf
        '''
        rewards[key] = BETA*np.exp(-1*distance(key,s_g)**2/(2*SIGMA**2))
    print("Rewards: {0}".format(rewards))
    beam.set(rewards, actions)

    # no policy initialization, we will derive our policy from most recent Q
    #initialize Q(s,a)
    Q = {}
    init_q=0.1
    states = beam.all_states()
    for s in states:
        Q[s] = {}
        for a in ALL_POSSIBLE_ACTIONS:
            Q[s][a] = init_q
            #Q[s][a] = BETA*(1+ 1/(1-GAMMA))*np.exp(-1*distance(s,s_g)**2/(2*SIGMA**2)) + INITQ_DELTA

    update_counts ={} #to keep track of how many times each state gonna get updated
    update_counts_sa={} #for adaptive learning rate
    for s in states:
        update_counts_sa[s] = {}
        for a in ALL_POSSIBLE_ACTIONS:
            update_counts_sa[s][a] = 1.0

    #repeat until converrgence
    t = 1.0
    deltas = []
    snr_lst=[]
    trail_lst=[]
    num_trails = 40000

    for iter in range(num_trails):
        if iter % 50 == 0:
            t += 1e-3
        if iter % 100 == 0:
            print("%d iterations are done" % iter)
        '''
        Setting parameter values to default for every iteration
        '''
        mimo.P_tx = 30 #default value
        mimo.AoA = 0 #default value

        s = start_state
        beam.set_state(s)
        a = (max_dict(Q[s]))[0]

        steps = 0
        biggest_change = 0
        while not beam.game_over(s):
            a = random_action(a, eps=1 / t)
            r = beam.move(a)
            s2 = beam.get_current_state()
            if beam.game_over(s2):
                break
            snr_lst.append(s2)

            alpha = ALPHA / update_counts_sa[s][a]
            update_counts_sa[s][a] += 0.005

            old_qsa = Q[s][a]
            a2, max_q_s2a2 = max_dict(Q[s2])
            #print("a2: {1}, Max qs2a2: {0}, Q[s][a]: {2}, r: {3}".format(max_q_s2a2, a2, Q[s][a], r))
            Q[s][a] = Q[s][a] + alpha*(r + GAMMA*max_q_s2a2 - Q[s][a])
            biggest_change = max(biggest_change, np.abs(old_qsa - Q[s][a]))

            update_counts[s] = update_counts.get(s,0) + 1
            steps+=1
            s = s2
            a = a2

        trail_lst.append(steps)
        deltas.append(biggest_change)
        '''
        print("Q: ")
        for s in ALL_POSSIBLE_STATES:
            for a in ALL_POSSIBLE_ACTIONS:
                print("s: {0}, a_ptx: {1}, a_aoa: {2}, ptx: {3}, AoA: {4}, AoD: {5}, Q: {6}".format(s,a[0], a[1], mimo.P_tx, mimo.AoA, mimo.AoD, Q[s][a]))
        #break
        '''
    print(snr_lst)
    print("Min: {0}, Max: {1}".format(min(snr_lst), max(snr_lst)))
    #print(dict((i, snr_lst.count(i)) for i in snr_lst))


    policy = {}
    V = {}
    for s in beam.actions.keys():
        a, max_q = max_dict(Q[s])
        policy[s] = a
        V[s] = max_q
        print("s: {0}, a_ptx: {1}, a_aoa: {2}, Q: {3}".format(s, a[0], a[1], V[s]))

    # what's the portion of time we spent updating each state in the grid

    print("Update counts: ")
    total = np.sum(list(update_counts.values()))
    for k, v in update_counts.items():
        # update_counts[k]
        update_counts[k] = float(v) / total
    print(update_counts)
    

    print(trail_lst)
    #plt.figure(1)
    plt.subplot(3,1,1)
    plt.plot(np.arange(num_trails)+1, trail_lst)
    plt.ylabel('No. of steps to reach goal')
    plt.xlabel('trail number')

    #plt.title('Rewards:beta*exp(-1*d(s,s_g)^2/(2*sigma^2))\n Q: beta*(1+ 1/(1-gamma))*np.exp(-1*distance(s,s_g)**2/(2*sigma**2)) + initq_delta\n beta:{1}, sigma:{2}, gamma:{0}, initq_delta:{3}'.format(GAMMA, BETA, SIGMA, INITQ_DELTA))
    plt.title('Rewards:beta*exp(-1*d(s,s_g)^2/(2*sigma^2))\n Q: {0}\n beta:{1}, sigma:{2}'.format(init_q, BETA, SIGMA))
    #plt.title('Rewards: {1} and {2}\n Q: {0}'.format(init_q, r_g, r_inf))
    #plt.show()


    '''
    Print the learnt optimal path in reaching the desired SNR
    '''
    start_state = 22
    max_state = np.max(ALL_POSSIBLE_STATES)
    learnt_mimo = MIMO(ALL_POSSIBLE_PwT_ACTIONS, ALL_POSSIBLE_AoA_ACTIONS)
    learnt_beam = Beam(ALL_POSSIBLE_STATES, learnt_mimo, start_state)
    learnt_beam.set(rewards, actions)
    SNR_path=[start_state]

    s = start_state
    loop_state= False
    while s < max_state:
        if len(SNR_path) > 20:
            loop_state = True
            break
        if s in policy.keys():
            a = policy[s]
            r = learnt_beam.move(a)
            s2 = learnt_beam.get_current_state()
            SNR_path.append(s2)
            s = s2

    if not loop_state:
        SNR_path.append(max_state)
    #plt.figure(2)
    plt.subplot(3,1,3)
    plt.plot(np.arange(len(SNR_path)) + 1, SNR_path, '.-')
    plt.axis([1,20,21,40])
    plt.ylabel('SNR states')
    plt.xlabel('Steps')
    plt.grid()
    '''
    plt.title(
        'Learnt Optimal path to desired SNR\n Rewards:beta*exp(-1*d(s,s_g)^2/(2*sigma^2))\n Q: beta*(1+ 1/(1-gamma))*np.exp(-1*distance(s,s_g)**2/(2*sigma**2)) + initq_delta\n beta:{1}, sigma:{2}, gamma:{0}, initq_delta:{3}'.format(
            GAMMA, BETA, SIGMA, INITQ_DELTA))
    '''
    #plt.title('Learnt optimal path to desired SNR\nRewards:beta*exp(-1*d(s,s_g)^2/(2*sigma^2))\n Q: {0}\n beta:{1}, sigma:{2}'.format(init_q, BETA, SIGMA))
    plt.title('Learnt optimal path to desired SNR')
    plt.show()

