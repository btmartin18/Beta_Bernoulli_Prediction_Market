"""
Script simulating instances of the Beta-Bernoulli market with varied data point sampling sequences
and trader sequences. Compensation is saved with each data point sequence paired with each trader
sequence.

Blake Martin    WN2021
"""

import numpy as np
import math
import os
import matplotlib.pyplot as plt
import random
from fractions import Fraction
import shutil

class Exp_agent:
    """
    Maintains information for a Bayesian trader with some amount of informativeness
    in an exponential family prediction market. Assumes linear utility.

    delta:          Vector of quantity of shares
    payment:        Sum of payments for each trade
    id:             A unique identifier
    obs_num:        Number of observations that this agent has access to [Informativeness]
    obs_noise:      Noise applied to the true data distribution when producing observed
                    examples. For Bernoulli, x ~ Bern(p + eps), eps ~ N(0, obs_noise). [Informativeness]
    budget:         Trader budget for budget-constrained trading
    data:           Data points this trader has access to
    """
    def __init__(self, name, obs_num=10, obs_noise=0, data=[], budget=float('inf')):
        self.delta = np.array([0., 0.])
        self.payment = 0
        self.id = name
        self.obs_num = obs_num
        self.obs_noise = obs_noise
        self.data = data

class Beta_Bernoulli_market():
    """
    Maintains information for the exponential family market maker

    shares:         Vector of outstanding shares
    shares_initial: Used to compute market loss (payoff - C(theta) + C(theta_initial))
    agents:         Dictionary mapping names to agents for agents who hold shares in the market
    """
    def __init__(self, p_true, shares_initial=np.array([0., 0.])):
        self.shares = np.array(shares_initial)
        self.shares_initial = np.array(shares_initial)
        self.agents = {}
        self.p_bounds = (0.05, 0.95)
        self.p_true = p_true

    def cost_fn(self, shares=None):
        """
        Cost(alpha, beta)
        """
        alpha, beta = self.shares + 1 if shares is None else shares + 1
        return math.log(math.factorial(alpha-1)) + math.log(math.factorial(beta-1)) \
               - math.log(math.factorial(alpha + beta - 1))

    def transaction(self, agent, wc_stats=False):
        """
        Model an agent making a transaction in this market
        """
        if agent.id not in self.agents.keys():
            self.agents[agent.id] = agent

        # Calculate posterior beliefs of trader
        private_sample = agent.data

        # Save previous market state then update shares
        shares_prev = [i for i in self.shares]
        cost_prev = self.cost_fn()
        self.shares += private_sample
        
        # Update agent's portfolio
        new_shares = self.shares - shares_prev
        new_pay = self.cost_fn() - cost_prev
        agent.delta += new_shares
        agent.payment += new_pay

        if wc_stats:
            comp = new_shares.dot(np.array([math.log(self.p_true), math.log(1 - self.p_true)])) - new_pay
            return [comp, shares_prev[0]+1, shares_prev[1]+1, new_shares, self.p_true]

    def simulate_rounds(self, agent_list, n_rounds=100, delta_shares=1e-6, shuffle=True, wc_stats=False):
        """
        Simulate n rounds of trading where in each round, each trader moves the price according to their belief.
        
        agent_list:     Agents who trade at each round
        n_rounds:       Maximum number of rounds where each agent is able to trade (convergence criteria)
        delta_shares:   Convergence critera for L2 norm of shares{t}-shares{t-1}
        plot_rounds:    Plot posterior at the end of each round in plot_rounds
        path:           Path to directory to save market summaries after each round in plot_rounds
        shuffle:        Boolean of whether to shuffle traders at each round
        """
        
        # Initialize price list
        share_1, share_2 = [], []
        share_1.append(self.shares[0])
        share_2.append(self.shares[1])

        worst_trade = [float('inf'), -1, -1, [], -1]

        for round_num in range(n_rounds):
            shares_orig = [i for i in self.shares]

            # Allow each agent to trade in random order
            if shuffle:
                random.shuffle(agent_list)
            for agent in agent_list:
                if wc_stats:
                    trade = self.transaction(agent, wc_stats)
                    if trade[0] < worst_trade[0]:
                        worst_trade = trade
                else:
                    self.transaction(agent)
                share_1.append(self.shares[0])
                share_2.append(self.shares[1])

        if wc_stats:
            return share_1, share_2, round_num, worst_trade
        else:
            return share_1, share_2, round_num


    def H(self, n):
        """
        Compute the nth harmonic number: sum{1/i} from i=1 to n
        """
        return sum(Fraction(1, d) for d in range(1, int(n)+1))

    def compute_distance(self, prev=None):
        """
        Compute the L2 distance between the expectation of the statistic phi with two different share vectors

        prev:   If prev is None, distance between current expectation of phi and the expected prices implied by p. Else
                the distance between expectation of phi with current vs previous share vector
        """
        if prev is None:
            diff_1 = self.H(self.shares[0]) - self.H(self.shares[0] + self.shares[1] + 1) - math.log(self.p_true)
            diff_2 = self.H(self.shares[1]) - self.H(self.shares[0] + self.shares[1] + 1) - math.log(1 - self.p_true)
            return math.sqrt(diff_1**2 + diff_2**2), (diff_1 + math.log(self.p_true), diff_2 + math.log(1 - self.p_true))
        else:
            diff_1 = self.H(self.shares[0]) - self.H(self.shares[0] + self.shares[1] + 1) - (self.H(prev[0]) - self.H(prev[0] + prev[1] + 1))
            diff_2 = self.H(self.shares[1]) - self.H(self.shares[0] + self.shares[1] + 1) - (self.H(prev[1]) - self.H(prev[0] + prev[1] + 1))
            return math.sqrt(diff_1**2 + diff_2**2)


def make_trader_list(seq, dp_list, high_info, low_info):
    """
    Make a list of agents (UB, varying informativeness) with a specified sequence

    seq:        sequence vector of length (# traders) where 1 corresponds to HI trader at
                that position and 0 corresponds to a LI trader at that position
    dp_list:    Data points list
    high_info:  Sample size of HI traders
    low_info:   Sample size of LI traders
    """
    agents = []
    dp = 0
    idx = 0

    for elt in seq:
        if elt == 1:
            points = dp_list[dp:dp+high_info]
            data = np.array([np.sum(points), high_info - np.sum(points)])
            agents.append(Exp_agent(name=str(idx), obs_num=high_info, data=data))
            dp += high_info
        else:
            points = dp_list[dp:dp+low_info]
            data = np.array([np.sum(points), low_info - np.sum(points)])
            agents.append(Exp_agent(name=str(idx), obs_num=low_info, data=data))
            dp += low_info
        idx += 1

    return agents


def make_confidence_interval(data_vector, rounding=None):
    mean = np.mean(data_vector, axis=0)
    std = np.std(data_vector, axis=0)
    N = len(data_vector)
    if rounding is None:
        return str(mean) + ' \\pm ' + str(1.96 * std / math.sqrt(N))
    elif type(mean) is list:
        return str([round(i*1000, rounding) for i in mean]) + ' \\pm ' + str([round(i*1000, rounding) for i in 1.96 * std / math.sqrt(N)])
    else:
        return str(round(mean*1000, rounding)) + ' \\pm ' + str(round(1960 * std / math.sqrt(N), rounding))


def main():

    # Informativeness
    n_low_info, n_high_info = 500, 500
    low_info, high_info = 1, 5

    n_datapoint_combs = 100
    n_seq_combs = 100

    for n_rounds in [1]:
        for p_true in [0.75, 0.5]:

            # Create directory
            path = 'randomness_simulations/' + str(p_true) + '/'
            if str(p_true) not in os.listdir('randomness_simulations/'):
                os.mkdir(path[:-1])

            arr_high_avg = np.zeros((n_datapoint_combs, n_seq_combs))
            arr_low_avg = np.zeros((n_datapoint_combs, n_seq_combs))
            arr_datapoints = np.zeros((n_datapoint_combs, n_low_info*low_info + n_high_info*high_info))
            arr_seq = np.zeros((n_seq_combs, n_low_info + n_high_info))

            for r in range(n_datapoint_combs):
                for c in range(n_low_info*low_info + n_high_info*high_info):
                    arr_datapoints[r, c] = np.random.binomial(1, p_true)

            for i in range(n_seq_combs):
                arr_seq[i, np.random.choice(np.arange(n_low_info + n_high_info), n_high_info, replace=False)] = 1

            for dp in range(n_datapoint_combs):
                for sq in range(n_seq_combs):
                    if sq%10 == 0:
                        print(str(p_true), str(dp), str(sq))

                    agents = make_trader_list(arr_seq[sq], arr_datapoints[dp], high_info, low_info)
                    market = Beta_Bernoulli_market(p_true)

                    share1, share2, rounds = market.simulate_rounds(agents, n_rounds, shuffle=False)

                    # Calculate compensation
                    phi = np.array([math.log(p_true), math.log(1 - p_true)])
                    exp_low = 0
                    exp_high = 0
                    for agent in agents:
                        exp = phi.dot(agent.delta) - agent.payment
                        if agent.obs_num == high_info:
                            exp_high += exp
                        else:
                            exp_low += exp

                    exp_high /= n_high_info
                    exp_low /= n_low_info
                    arr_high_avg[dp, sq] = exp_high
                    arr_low_avg[dp, sq] = exp_low

            print('HI avg:', str(np.mean(arr_high_avg)))
            print('LI avg:', str(np.mean(arr_low_avg)))

            with open(path + 'high_comp.txt', 'w') as f:
                for r in arr_high_avg:
                    for c in r:
                        f.write(str(c) + ' ')
                    f.write('\n')

            with open(path + 'low_comp.txt', 'w') as f:
                for r in arr_low_avg:
                    for c in r:
                        f.write(str(c) + ' ')
                    f.write('\n')

            with open(path + 'points.txt', 'w') as f:
                for r in arr_datapoints:
                    for c in r:
                        f.write(str(int(c)) + ' ')
                    f.write('\n')

            with open(path + 'sequences.txt', 'w') as f:
                for r in arr_seq:
                    for c in r:
                        f.write(str(int(c)) + ' ')
                    f.write('\n')

if __name__ == '__main__':
    main()