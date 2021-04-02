"""
Script simulating many instances of the Beta-Bernoulli market for UB traders
with either high or low informativeness. Results are saved to the info_simulations/ folder

Blake Martin    WN2021
"""

import numpy as np
from beta_bernoulli import *
import math
import shutil

def make_confidence_interval(data_vector, rounding=None):
    """
    95% confidence interval
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

## Parameters that are consistent between runs ##
n_sims = 1000
n_entries = 1 # No repeated trading within an instance of the market

# Informativeness
n_low_info, n_high_info = 500, 500
low_info, high_info = 1, 5

# Convergence
delta_price = 0 # Not used for these runs

# p_true drawn from uniform distribution or constant
p_true = []
for s in range(n_sims):
    # p_true.append(np.random.uniform(0.05, 0.95))
    p_true.append(0.75)
p_true = np.array(p_true)

for seq in ['high_first', 'low_first', 'interleaved']:

    # Create directory
    path = 'info_simulations/' + seq + '/'
    if seq not in os.listdir('info_simulations/'):
        os.mkdir(path[:-1])

    arr_high_avg = []
    arr_low_avg = []
    arr_converged_params = []
    arr_params = np.zeros((n_low_info + n_high_info + 1, 2))
    arr_comp = np.zeros(n_low_info + n_high_info)
    wc_stats = True
    wc_trade = [float('inf'), -1, -1, [], -1] # comp, alpha, beta, sample, p_true
    wc_trade_avg = []

    # Perform n_sims simulations of the market to convergence
    for sim in range(n_sims):
        if sim%10 == 0:
            print(seq + ' ' + str(round(p_true[sim], 2)) + ' ' + str(sim))

        agents, shuffle = make_trader_list(seq, n_high_info, n_low_info, high_info, low_info)
        market = Beta_Bernoulli_market(p_true[sim])

        if not wc_stats:
            share1, share2, entries = market.simulate_trades(agents, n_entries, delta_price, shuffle)
        else:
            share1, share2, entries, worst_trade = market.simulate_trades(agents, n_entries, delta_price, shuffle, wc_stats)
            wc_trade_avg.append(worst_trade[0])
            if worst_trade[0] < wc_trade[0]:
                wc_trade = worst_trade

        # Calculate average parameters
        shares = np.transpose(np.vstack((share1, share2))) + 1
        arr_params += shares / (n_sims)

        # Calculate compensation
        phi = np.array([math.log(p_true[sim]), math.log(1 - p_true[sim])])
        exp_low = 0
        exp_high = 0
        inner_arr_comp = []
        for agent in agents:
            exp = phi.dot(agent.delta) - agent.payment
            inner_arr_comp.append(exp)
            if agent.obs_num == high_info:
                exp_high += exp
            else:
                exp_low += exp

        exp_high /= n_high_info
        exp_low /= n_low_info

        arr_converged_params.append([share1[-1]+1, share2[-1]+1])
        arr_high_avg.append(exp_high)
        arr_low_avg.append(exp_low)
        arr_comp += np.array(inner_arr_comp) / n_sims

    with open(path + 'params.txt', 'w') as share_f:
        share_f.write('[alpha, beta]\n')
        for param in arr_params:
            share_f.write(str(param) + '\n')

    with open(path + 'compensation.txt', 'w') as comp_f:
        for comp in arr_comp:
            comp_f.write(str(comp) + '\n')

    # Here, we look at how the first/last 10 of each type of trader was compensated
    if seq == 'high_first':
        first_10_low = arr_comp[n_high_info:n_high_info+10]
        last_10_low = arr_comp[-10:]
        first_10_high = arr_comp[:10]
        last_10_high = arr_comp[n_high_info-10: n_high_info]
    elif seq == 'low_first':
        first_10_high = arr_comp[n_low_info:n_low_info+10]
        last_10_high = arr_comp[-10:]
        first_10_low = arr_comp[:10]
        last_10_low = arr_comp[n_low_info-10:n_low_info]
    elif seq == 'interleaved':
        first_10_low = arr_comp[0:20:2]
        last_10_low = arr_comp[-20::2]
        first_10_high = arr_comp[1:20:2]
        first_10_high = arr_comp[-19::2]

    with open(path + 'summary.txt', 'w') as f:
        f.write('Sequence type:               ' + seq)
        f.write('\nHigh informativeness:        ' + str(n_high_info) + ' traders with ' + str(high_info) + ' observations')
        f.write('\nLow informativeness:         ' + str(n_low_info) + ' traders with ' + str(low_info) + ' observations')
        f.write('\nConvergence criteria:        ' + str(n_entries) + ' trades per trader or ' + str(delta_price) + ' change in L2 norm of price vector')

        f.write('\n\n\nCONFIDENCE INTERVALS:')
        f.write('\n\nConverged Params:           ' + make_confidence_interval(arr_converged_params))
        f.write('\n\nAverage High Info Comp:     ' + make_confidence_interval(arr_high_avg))
        f.write('\n\nAverage Low Info Comp:      ' + make_confidence_interval(arr_low_avg))
        f.write('\n\nFirst 10 low:               ' + make_confidence_interval(first_10_low))
        f.write('\n\nFirst 10 high:              ' + make_confidence_interval(first_10_high))
        f.write('\n\nLast 10 low:                ' + make_confidence_interval(last_10_low))
        f.write('\n\nLast 10 high:               ' + make_confidence_interval(last_10_high))
        f.write('\n\nGlobal WC Trade:            ' + str(wc_trade))
        f.write('\n\nWC Trade Comp:              ' + make_confidence_interval(wc_trade_avg))


    # For more quickly putting information into latex tables
    with open('info_simulations/table.txt', 'a') as f:
        d = {
            'high_first': 'High Info First',
            'low_first': 'Low Info First',
            'interleaved': 'Interleaved'
        }
        r = 2
        f.write(d[seq] + ' & $' + make_confidence_interval(arr_high_avg, r) + '$ & $' + make_confidence_interval(arr_low_avg, r) + '$ \\\\\n')


    with open('info_simulations/ends_table.txt', 'a') as f:
        d = {
            'high_first': 'High Info First',
            'low_first': 'Low Info First',
            'interleaved': 'Interleaved'
        }
        r = 2
        f.write(d[seq] + ' & $' + make_confidence_interval(first_10_low, r) + \
            ' & $' + make_confidence_interval(first_10_high, r) + \
            ' & $' + make_confidence_interval(last_10_low, r) + \
            '$ & $' + make_confidence_interval(last_10_high, r) + '$ \\\\\n')