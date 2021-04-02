"""
Script simulating many instances of the Beta-Bernoulli market for traders
with either limited or unlimited budgets. Results are saved to the budget_simulations/ folder

Blake Martin    WN2021
"""

import numpy as np
from beta_bernoulli import *
import math
import shutil

def make_confidence_interval(data_vector, rounding=None):
    """
    95% Confidence interval
    """
    mean = np.mean(data_vector, axis=0)
    std = np.std(data_vector, axis=0)
    N = len(data_vector)
    if rounding is None:
        return str(mean) + ' \\pm ' + str(1.96 * std / math.sqrt(N))
    elif type(mean) is list:
        return str([round(i*1000, rounding) for i in mean]) + ' \\pm ' + str([round(i*1000, rounding) for i in 1.96 * std / math.sqrt(N)])
    else:
        return str(round(mean*1000, rounding)) + ' \\pm ' + str(round(1960 * std / math.sqrt(N), rounding))

def save_info(fname, market, diff, exp_prices, shares):
    """
    Saves relevant information about a specific instance of a market
    """
    with open(fname, 'w') as f:
        f.write('diff ' + str(diff))
        f.write('\n\np_true ' + str(market.p_true))
        f.write('\nE[p]     ' + str((market.shares[0]+1) / (market.shares[0] + market.shares[1]+2)))
        f.write('\ntrue prices ' + str(math.log(market.p_true)) + ' ' + str(math.log(1 - market.p_true)))
        f.write('\nexp  prices ' + str(exp_prices[0]) + ' ' + str(exp_prices[1]))
        f.write('\n\nbeta params ' + str(market.shares[0]+1) +' '+ str(market.shares[1]+1))
        f.write('\n\n')
        for share in shares:
            f.write('\n' + str(share[0]) + ' ' + str(share[1]))


## Parameters that are consistent between runs ##
n_sims = 1000
n_instances = 25 # Rounds of trading (with outcome/payoffs) per simulation
n_entries = 60 # Max number of times a trader can trade per round/instance

# budget
info = 5
n_lim, n_inf = 5, 5
budget = 10

# Convergence
delta_price = 0.01

# p_true drawn from uniform distribution
p_true = []
for s in range(n_sims):
    inner = []
    for i in range(n_instances):
        p = np.random.uniform(0.05, 0.95)
        inner.append(p)
    p_true.append(inner)
p_true = np.array(p_true)

for seq in ['low_first', 'high_first', 'interleaved']:

    # Create directory
    path = 'budget_simulations/' + seq + '/'
    if seq not in os.listdir('budget_simulations/'):
        os.mkdir(path[:-1])

    # General information gatherers
    arr_high_avg = []
    arr_low_avg = []
    arr_converged_params = []
    arr_budget = []
    arr_pts = []
    arr_rounds = []

    # arr_pts_k refers to tot points used by each LB trader at market instance k
    # div_k is the denominator for calculating the average number of points used at each trade
    arr_pts_1 = np.zeros(n_lim*n_entries)
    arr_pts_10 = np.zeros(n_lim*n_entries)
    arr_pts_25 = np.zeros(n_lim*n_entries)
    div_1, div_10, div_25 = np.zeros(n_lim*n_entries), np.zeros(n_lim*n_entries), np.zeros(n_lim*n_entries)

    # diffsk is the best and worst L2 distance between expectation of prices given current shares vs implied by p at market instance k
    diffs1 = [float('inf'), -1.0]
    diffs10 = [float('inf'), -1.0]
    diffs25 = [float('inf'), -1.0]
    arr_diff = np.zeros(n_instances) # averages

    # Perform n_sims simulations of the market to convergence
    for sim in range(n_sims):
        if sim%10 == 0:
            print(seq + ' ' + str(sim))

        agents, shuffle = budget_trader_list(seq, n_inf, n_lim, info, budget)

        converged_shares = []
        inner_rounds = []
        for market_instance in range(n_instances):

            market = Beta_Bernoulli_market(p_true[sim, market_instance])
            share1, share2, rounds = market.simulate_trades(agents, n_entries, delta_price, shuffle)
            inner_rounds.append(rounds)

            converged_shares.append([share1[-1]+1, share2[-1]+1])
            shares = np.transpose(np.vstack((share1, share2))) + 1

            # Save information about the best and worst market at instance 1, 10, 25
            dist, exp_prices = market.compute_distance()
            arr_diff[market_instance] += dist / n_sims
            if market_instance == 0:
                if dist < diffs1[0]:
                    diffs1[0] = dist
                    save_info(path + 'best1.txt', market, dist, exp_prices, shares)
                if dist > diffs1[1]:
                    diffs1[1] = dist
                    save_info(path + 'worst1.txt', market, dist, exp_prices, shares)
            elif market_instance == 10:
                if dist < diffs10[0]:
                    diffs10[0] = dist
                    save_info(path + 'best10.txt', market, dist, exp_prices, shares)
                if dist > diffs10[1]:
                    diffs10[1] = dist
                    save_info(path + 'worst10.txt', market, dist, exp_prices, shares)
            elif market_instance == 24:
                if dist < diffs25[0]:
                    diffs25[0] = dist
                    save_info(path + 'best25.txt', market, dist, exp_prices, shares)
                if dist > diffs25[1]:
                    diffs25[1] = dist
                    save_info(path + 'worst25.txt', market, dist, exp_prices, shares)

            # Save information about evolution of # points used
            idx = -1
            for trader in agents:
                if not trader.budget == float('inf'):
                    idx += 1
                    if len(trader.num_points) != n_entries:
                        trader.num_points = np.hstack((trader.num_points, np.zeros(60 - len(trader.num_points))))
                    addition = np.zeros(n_lim*n_entries)
                    addition[idx::n_lim] += trader.num_points
                    if market_instance == 0:
                        arr_pts_1 += addition
                        div_1[:n_lim*(rounds+1)] += (1/5)
                    elif market_instance == 9:
                        arr_pts_10 += addition
                        div_10[:n_lim*(rounds+1)] += (1/5)
                    elif market_instance == 24:
                        arr_pts_25 += addition
                        div_25[:n_lim*(rounds+1)] += (1/5)
                trader.compensate([math.log(p_true[sim, market_instance]), math.log(1-p_true[sim, market_instance])])

        # Calculate compensation
        exp_low = 0
        exp_high = 0
        for agent in agents:
            if agent.budget == float('inf'):
                exp_high += np.sum(agent.comp_tracker)
            else:
                exp_low += np.sum(agent.comp_tracker)
                arr_budget.append(np.array(agent.budget_tracker))
                arr_pts.append(np.array(agent.points_used))
        exp_high /= n_inf
        exp_low /= n_lim

        arr_converged_params.append(converged_shares)
        arr_high_avg.append(exp_high)
        arr_low_avg.append(exp_low)
        arr_rounds.append(inner_rounds)

    arr_pts_1 /= div_1
    arr_pts_10 /= div_10
    arr_pts_25 /= div_25

    with open(path + 'points1.txt', 'w') as f:
        for i in arr_pts_1:
            f.write(str(i) + '\n')

    with open(path + 'points10.txt', 'w') as f:
        for i in arr_pts_10:
            f.write(str(i) + '\n')

    with open(path + 'points25.txt', 'w') as f:
        for i in arr_pts_25:
            f.write(str(i) + '\n')

    with open(path + 'summary.txt', 'w') as f:
        f.write('Sequence type:               ' + seq)
        f.write('\nHigh Budget:                 ' + str(n_inf) + ' traders with ' + str(info) + ' observations and inf budget')
        f.write('\nLow Budget:                  ' + str(n_lim) + ' traders with ' + str(info) + ' observations and ' + str(budget) + ' budget')
        f.write('\nConvergence criteria:        ' + str(n_entries) + ' trades per trader or ' + str(delta_price) + ' change in L2 norm of price vector')

        f.write('\n\n\nCONFIDENCE INTERVALS:')
        f.write('\n\nConverged Params:           ' + make_confidence_interval(arr_converged_params))
        f.write('\n\nAverage High Budget Comp:     ' + make_confidence_interval(arr_high_avg))
        f.write('\n\nAverage Low Budget Comp:      ' + make_confidence_interval(arr_low_avg))
        f.write('\n\nBudget for limited:           ' + make_confidence_interval(arr_budget))
        f.write('\n\nPoints Used:                  ' + make_confidence_interval(arr_pts))
        f.write('\n\nRounds:                       ' + make_confidence_interval(arr_rounds))
        f.write('\n\nAvg diff:                     ' + str(arr_diff))

    # For more quickly putting information into latex tables
    with open('budget_simulations/table.txt', 'a') as f:
        d = {
            'high_first': 'High Budget First',
            'low_first': 'Low Budget First',
            'interleaved': 'Interleaved'
        }
        r=2
        f.write(d[seq] + ' & $' + make_confidence_interval(np.array(arr_high_avg), r) + '$ & $' + make_confidence_interval(np.array(arr_low_avg), r) + '$ \\\\\n')


