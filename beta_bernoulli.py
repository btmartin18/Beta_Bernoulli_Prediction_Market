"""
Classes and functions for creating a market and agents in a Beta-Bernoulli market
    - class Exp_agent
    - class Beta_Bernoulli_market
    - fn make_trader_list
    - fn budget_trader_list

Blake Martin    WN2021
"""

import numpy as np
import math
import os
import random
from fractions import Fraction

class Exp_agent:
    """
    Maintains information for a Bayesian trader with some amount of informativeness
    in an exponential family prediction market. Assumes linear utility, myopic traders.

    delta:              Vector of quantity of shares
    payment:            Sum of payments for each trade
    id:                 A unique identifier
    obs_num:            Number of observations that this agent has access to [Informativeness]
    obs_noise:          Noise applied to the true data distribution when producing observed
                        examples. For Bernoulli, x ~ Bern(p + eps), eps ~ N(0, obs_noise). [Informativeness]
    budget:             Trader budget for budget-constrained trading
    comp_tracker:       Tracks agent's compensation at each round of the market
    budget_tracker:     Tracks agent's budget at each round of the market
    prev:               For a budget-limited trade, saves the most recent data point this agent was
                        unable to trade on
    num_points:         Tracks the number of data points the trader used at each trade within one market round
    points_used:        Tracks the total number of data points the trader used within one market round
    """
    def __init__(self, name, obs_num=10, obs_noise=0, budget=float('inf')):
        self.delta = np.array([0., 0.])
        self.payment = 0
        self.id = name
        self.obs_num = obs_num
        self.obs_noise = obs_noise
        self.budget = budget
        self.comp_tracker = []
        self.budget_tracker = [budget]
        self.prev = None
        self.num_points = []
        self.points_used = []

    def compensate(self, phi):
        """
        After one round of the market, compensates traders by updating
        budget, resets shares and payment accumulated, and updates tracker variables

        phi:        payoff vector
        """
        U = self.delta.dot(phi) - self.payment
        self.budget += U
        self.payment = 0
        self.delta = np.array([0., 0.])
        self.comp_tracker.append(U)
        self.budget_tracker.append(self.budget)
        self.points_used.append(np.sum(self.num_points))
        self.num_points = []
        self.prev = None

class Beta_Bernoulli_market():
    """
    Maintains information for the Beta-Bernoulli market maker

    shares:             Vector of outstanding shares
    shares_initial:     Used to compute market loss (payoff - C(theta) + C(theta_initial))
    agents:             Dictionary mapping names to agents for agents who hold shares in the market
    p_bounds:           Extremes of success probability parameter. Provides bounds on agent loss
    p_true:             True value of success probability parameter of the Bernoulli data distribution
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

    def price_fn(self, security):
        """
        Price(alpha, beta) for a specified security
            - security \in {'alpha', 'beta'}
        """
        if security == 'alpha':
            return self.H(self.shares[0]) - self.H(self.shares[0] + self.shares[1] + 1)
        elif security == 'beta':
            return self.H(self.shares[1]) - self.H(self.shares[0] + self.shares[1] + 1)
        else:
            assert False, 'Security argument must be \\in {\'alpha\', \'beta\'}'

    def check_if_valid(self, agent, full_sample, sub_length):
        """
        Evaluates whether a limited budget agent is able to trade on a subset of their data points

        agent:          Trader making a transaction
        full_sample:    All of the data points a trader could have access to at this trade
        sub_length:     Subset length of the full sample that is evaluated
        """
        sum_sample = np.sum(full_sample[:sub_length])
        share_change = np.array([sum_sample, sub_length - sum_sample])

        share_ahead = self.shares + share_change
        payment = self.cost_fn(share_ahead) - self.cost_fn()

        payoff1 = np.array([math.log(self.p_bounds[0]), math.log(1-self.p_bounds[0])]).dot(agent.delta + share_change)
        payoff2 = np.array([math.log(self.p_bounds[1]), math.log(1-self.p_bounds[1])]).dot(agent.delta + share_change)
        worst_payoff = min(payoff1, payoff2)

        U_min = worst_payoff - agent.payment - payment
        return True if U_min >= -1 * agent.budget else False

    def sample_true_dist(self, agent):
        """
        Sample from the true distribution and return sum{phi(x)} = [sum{x}, sum{1-x}]

        agent:          Trader making a transaction
        """
        # Randomly sample the noise by eps ~ N(0, obs_noise)
        eps = np.random.normal(0, agent.obs_noise)
        eps = 1 if self.p_true + eps > 1 else 0 if self.p_true + eps < 0 else eps

        if agent.budget == float('inf'):
            # Sample from the (possibly noisy) true data distribution and return sum{phi(x)}
            sum_sample_data = np.random.binomial(agent.obs_num, self.p_true + eps)
            return np.array([sum_sample_data, agent.obs_num - sum_sample_data])

        else:
            # Trader is budget-limited. Sequentially check if they can trade on each data point
            if agent.prev is None:
                sample = [np.random.binomial(1, self.p_true + eps) for _ in range(agent.obs_num)]
            else:
                sample = agent.prev + [np.random.binomial(1, self.p_true + eps) for _ in range(agent.obs_num-len(agent.prev))]
            return_len = len(sample)
            agent.prev = None
            for sample_len in range(1, len(sample)+1):
                if not self.check_if_valid(agent, sample, sample_len):
                    return_len = sample_len - 1
                    agent.prev = sample[return_len:]
                    break
            sum_sample = np.sum(sample[:return_len])
            agent.num_points.append(return_len)
            return np.array([sum_sample, return_len - sum_sample])

    def change_price_convergence(self, share_prev, delta_price):
        """
        Check if our convergence criteria has been met. We use the distance between the prices after each agent has
        traded t times and after each trader has traded (t-1) times. Only relevant for market with repeated trades

        share_prev:         Suppose each trader has traded t times. This would be the share vector after each
                            trader has been given the opportunity to trade (t-1) times.
        delta_price:        We have converged if distance > delta_price.
        """
        change = self.compute_distance(prev=share_prev)
        return True if change < delta_price else False

    def transaction(self, agent, wc_stats=False):
        """
        Model an agent making a transaction in this market

        agent:          Trader making a transaction
        wc_stats:       If true, returns information describing the worst compensation made from a trade in this round
        """
        if agent.id not in self.agents.keys():
            self.agents[agent.id] = agent

        # Calculate posterior beliefs of trader
        private_sample = self.sample_true_dist(agent)

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

    def simulate_trades(self, agent_list, n_entries=100, delta_price=1e-6, shuffle=True, wc_stats=False):
        """
        Simulate a market to convergence
        
        agent_list:         Agents who trade at each round
        n_entries:          Maximum number of times each trader enters the market (convergence criterion #1)
        delta_price:        Convergence criterion of L2 norm of price{t}-price{t-1} (convergence criterion #2)
        plot_rounds:        Plot posterior at the end of each round in plot_rounds
        path:               Path to directory to save market summaries after each round in plot_rounds
        shuffle:            Boolean of whether to shuffle traders at each round
        wc_stats:           If true, returns information describing the worst compensation made from a trade in this round
        """
        
        # Initialize price list
        share_1, share_2 = [], []
        share_1.append(self.shares[0])
        share_2.append(self.shares[1])

        worst_trade = [float('inf'), -1, -1, [], -1]

        for round_num in range(n_entries):
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

            # Check delta_theta convergence criteria
            conv = self.change_price_convergence(shares_orig, delta_price)
            if conv:
                break

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


def make_trader_list(seq, n_high_info, n_low_info, high_info, low_info):
    """
    Make a list of agents (UB, varying informativeness) with a specified sequence

    seq:            Trader sequence as 'high_first', 'low_first', 'interleaved', 'all_high', 'all_low', or 'shuffle'
    n_high_info:    Number of HI traders
    n_low_info:     Number of LI traders
    high_info:      Sample size of HI traders
    low_info:       Sample size of LI traders
    """
    agents = []
    shuffle = False

    # High information traders enter first
    if seq == 'high_first' or seq == 'shuffle':
        for trader_num in range(n_high_info):
            agents.append(Exp_agent(name=str(trader_num), obs_num=high_info))
        start_num = n_high_info
        for trader_num in range(n_low_info):
            agents.append(Exp_agent(name=str(trader_num + start_num), obs_num=low_info))

    # Low information traders enter first
    elif seq == 'low_first':
        for trader_num in range(n_low_info):
            agents.append(Exp_agent(name=str(trader_num), obs_num=low_info))
        start_num = n_low_info
        for trader_num in range(n_high_info):
            agents.append(Exp_agent(name=str(trader_num + start_num), obs_num=high_info))

    # Interleave high and low information traders
    elif seq == 'interleaved':
        assert n_low_info == n_high_info, 'Number of low and high information traders should be equal for interleaving'
        for trader_num in range(n_low_info):
            agents.append(Exp_agent(name=str(trader_num*2), obs_num=low_info))
            agents.append(Exp_agent(name=str(trader_num*2 + 1), obs_num=high_info))

    elif seq == 'all_high':
        for trader_num in range(n_high_info):
            agents.append(Exp_agent(name=str(trader_num), obs_num=high_info))

    elif seq == 'all_low':
        for trader_num in range(n_low_info):
            agents.append(Exp_agent(name=str(trader_num), obs_num=low_info))

    # Shuffle at each round
    elif seq == 'shuffle':
        shuffle = True

    else:
        print('Provide a valid seq')
        quit()

    return agents, shuffle


def budget_trader_list(seq, n_high_budget, n_low_budget, info, budget):
    """
    Make a list of agents (constant info, varying budget) with a specified sequence

    seq:            Trader sequence as 'high_first', 'low_first', 'interleaved', 'all_high', 'all_low', or 'shuffle'
    n_high_budget:  Number of UB traders
    n_low_budget:   Number of LB traders
    info:           Sample size of traders
    budget:         Budget of LB traders
    """
    agents = []
    shuffle = False

    # High information traders enter first
    if seq == 'high_first' or seq == 'shuffle':
        for trader_num in range(n_high_budget):
            agents.append(Exp_agent(name=str(trader_num), obs_num=info))
        start_num = n_high_budget
        for trader_num in range(n_low_budget):
            agents.append(Exp_agent(name=str(trader_num + start_num), obs_num=info, budget=budget))

    # Low information traders enter first
    elif seq == 'low_first':
        for trader_num in range(n_low_budget):
            agents.append(Exp_agent(name=str(trader_num), obs_num=info, budget=budget))
        start_num = n_low_budget
        for trader_num in range(n_high_budget):
            agents.append(Exp_agent(name=str(trader_num + start_num), obs_num=info))

    # Interleave high and low information traders
    elif seq == 'interleaved':
        assert n_low_budget == n_high_budget, 'Number of low and high information traders should be equal for interleaving'
        for trader_num in range(n_low_budget):
            agents.append(Exp_agent(name=str(trader_num*2), obs_num=info, budget=budget))
            agents.append(Exp_agent(name=str(trader_num*2 + 1), obs_num=info))

    elif seq == 'all_high':
        for trader_num in range(n_high_budget):
            agents.append(Exp_agent(name=str(trader_num), obs_num=info))

    elif seq == 'all_low':
        for trader_num in range(n_low_budget):
            agents.append(Exp_agent(name=str(trader_num), obs_num=info, budget=budget))

    # Shuffle at each round
    elif seq == 'shuffle':
        shuffle = True

    else:
        print('Provide a valid seq')
        quit()

    return agents, shuffle