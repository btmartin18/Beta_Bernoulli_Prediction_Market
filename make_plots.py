"""
Functions for create plots using the information output by budget_vary.py and
informativeness_vary.py

Blake Martin    WN2021
"""

import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.stats import beta

def read_data(fname):
    """
    Converts file of alpha, beta parameters into a numpy array

    fname: file name with alpha, beta data
    """
    params = []
    with open(fname, 'r') as f:
        lines = f.readlines()
        for line in lines[1:]:
            inner_list = []
            for i in line[1:-2].split():
                if len(i):
                    inner_list.append(float(i))
            params.append(inner_list)
    return np.array(params)


def split_string(text):
    """
    Splits the confidence intervals from summary.txt into avg and std arrays

    text: line containing the confidence intervals
    """
    first = text.find('[')
    end_first = text.find(']')
    second = text.find('[', end_first+1)
    end_second = text.find(']', end_first+1)

    avg = [float(num) for num in text[first+1:end_first].split()]
    std = [float(num) for num in text[second+1:end_second].split()]

    return np.array(avg), np.array(std)


def beta_plot(params):
    """
    Given alpha and beta, produce the beta pdf plot data

    params: [alpha, beta]
    """
    theta = np.linspace(0.01, 0.99, 495)
    post_pdf = beta.pdf(theta, params[0], params[1])
    return theta, post_pdf


def make_info_plots(direc):
    """
    Make plots from the information output by informativeness_vary.py
    Includes exp[p] plot and beta plots

    direc: directory (generally info_simulations/) containing the outputs with different sequences
    """
    fs = 16
    d = {
        'high_first': 'HI First',
        'low_first': 'LI First',
        'interleaved': 'Interleaved'
    }

    for p_true in [0.75, 0.5, 0.25]:
        fig_exp_p, ax_exp_p = plt.subplots(dpi=200)
        fig, ((ax11, ax12), (ax21, ax22), (ax31, ax32)) = plt.subplots(nrows=3, ncols=2, dpi=200)
        for seq in ['high_first', 'low_first', 'interleaved']:
            path = direc + str(p_true) + '/' + seq + '/'
            params = read_data(path + 'params.txt')

            # Expectation of p plot
            exp_p = []
            for param in params:
                exp_p.append(param[0] / (param[0] + param[1]))
            ax_exp_p.plot(np.arange(0, 1001)+1, exp_p, label=d[seq])
            ax_exp_p.set_xlabel('# Trades')
            ax_exp_p.set_ylabel('Expectation of $p$')

            # Beta plots
            mid_x, mid_y = beta_plot(params[500, :])
            end_x, end_y = beta_plot(params[1000, :])
            if seq[0] == 'h':
                ax11.set_title('HI First: 500 Trades')
                ax12.set_title('HI First: 1000 Trades')
                ax11.plot(mid_x, mid_y)
                ax12.plot(end_x, end_y)
            elif seq[0] == 'l':
                ax21.set_title('LI First: 500 Trades')
                ax22.set_title('LI First: 1000 Trades')
                ax21.plot(mid_x, mid_y)
                ax22.plot(end_x, end_y)
            else:
                ax31.set_title('Interleaved: 500 Trades')
                ax32.set_title('Interleaved: 1000 Trades')
                ax31.plot(mid_x, mid_y)
                ax32.plot(end_x, end_y)
        for ax in [ax11, ax12, ax21, ax22, ax31, ax32]:
            ax.set_xlabel('p')
            ax.set_ylabel('Probability Density')
            ax.set_ylim([0, 50])

        ax_exp_p.legend()
        ax_exp_p.set_xscale('log')
        fig.tight_layout()
        fig.savefig(direc + str(p_true) + '/beta_plot.png')
        fig_exp_p.savefig(direc + str(p_true) + '/exp_p.png')


def make_n_plots(direc, nrange):
    """
    Given the low_first sequence, plot where LI compensation intersects
    HI compensation for varying n traders
    """
    fig, ax = plt.subplots(dpi=200)
    high_comp = []
    low_comp = []
    for n in nrange:
        fig_exp_p, ax_exp_p = plt.subplots(dpi=200)
        path = direc + str(n) + '/'
        params = read_data(path + 'params.txt')

        # Expectation of p plot
        exp_p = []
        for param in params:
            exp_p.append(param[0] / (param[0] + param[1]))
        ax_exp_p.plot(np.arange(0, len(exp_p)), exp_p)
        ax_exp_p.set_xlabel('# Trades')
        ax_exp_p.set_ylabel('Expectation of $p$')

        with open(path + 'summary.txt')as f:
            lines = f.readlines()
            high_comp.append(float(lines[11][28:47]))
            low_comp.append(float(lines[12][28:47]))


        # ax_exp_p.set_xscale('log')
        fig_exp_p.savefig(direc + str(n) + '/exp_p.png')

    ax.plot(nrange, low_comp)
    ax.plot(nrange, high_comp)
    ax.legend(['LI Traders', 'HI Traders'])
    ax.set_xlabel('# Traders of Each Type')
    ax.set_ylabel('Average Compensation')
    fig.savefig(direc+'compensation.png')


def make_budget_plots(direc, seqs=['high_first', 'low_first', 'interleaved']):
    """
    For a directory corresponding to a LB trader budget, plots
    budget vs market round for each given sequence
    """
    fig, ax = plt.subplots(dpi=200)
    d = {
        'high_first': 'UB First',
        'low_first': 'LB First',
        'interleaved': 'Interleaved'
    }
    for seq in seqs:
        path = direc + seq + '/'
        with open(path + 'summary.txt', 'r') as f:
            read = False
            out_str = ''
            for line in f:
                if line[:6] == 'Budget':
                    read = True
                if line[:6] == 'Points':
                    read = False
                if read:
                    out_str += line
        avg, std = split_string(out_str)
        time_steps = np.arange(0, len(avg))
        ax.plot(time_steps, avg, label=d[seq])
        ax.fill_between(time_steps, (avg - std), (avg + std), alpha=0.1)
        ax.set_xlabel('Market Round')
        ax.set_ylabel('Avg Budget of LB Traders')
    ax.legend()
    fig.savefig(direc + 'budget_plot.png')


def make_many_budget_plots(direcs=None, seqs=['high_first', 'low_first', 'interleaved']):
    """
    For each directory, makes budget plots and number of datapoints used plots
    """
    if direcs is None:
        #direcs = ['budget_2/', 'budget_5/', 'budget_10/', 'budget_100/', 'budget_1000000/']
        direcs = ['budget_10/']
    for direc in direcs:
        make_budget_plots(direc, seqs)
        fsize = (15, 10)
        fig_first, ((ax_low1, ax_high1, ax_inter1)) = plt.subplots(nrows=3, ncols=1, dpi=200)
        fig_last, ((ax_low5, ax_high5, ax_inter5)) = plt.subplots(nrows=3, ncols=1, dpi=200)
        fig_first, ((ax_low1, ax_high1)) = plt.subplots(nrows=2, ncols=1, dpi=200)
        fig_last, ((ax_low5, ax_high5)) = plt.subplots(nrows=2, ncols=1, dpi=200)
        # fig_first.suptitle('First LB Trader')
        # fig_last.suptitle('Final LB Trader')
        ax_low1.set_title('LB Traders First')
        ax_low5.set_title('LB Traders First')
        ax_high1.set_title('UB Traders First')
        ax_high5.set_title('UB Traders First')
        ax_inter1.set_title('Interleaved')
        ax_inter5.set_title('Interleaved')

        for seq in seqs:
            if seq == 'low_first':
                ax1 = ax_low1
                ax5 = ax_low5
            if seq == 'high_first':
                ax1 = ax_high1
                ax5 = ax_high5
            if seq == 'interleaved':
                ax1 = ax_inter1
                ax5 = ax_inter5
            # Make points plot
            for instance in [1, 10, 25]:
                with open(direc + seq.lower() + '/points' + str(instance) + '.txt', 'r') as f:
                    arr = []
                    for line in f:
                        if float(line) is not np.nan:
                            arr.append(float(line))
                    arr1 = arr[0::5]
                    arr5 = arr[4::5]
                    ax1.plot(np.arange(1, len(arr1)+1), arr1, label='Round = ' + str(instance))
                    ax5.plot(np.arange(1, len(arr5)+1), arr5, label='Round = ' + str(instance))
            if direc[-3:] == '10/' or direc[-2:] == '5/':
                ax1.set_xlim([1, 10])
                ax5.set_xlim([1, 10])
                ax1.set_xticks([2, 4, 6, 8, 10])
                ax5.set_xticks([2, 4, 6, 8, 10])
            elif direc[-2:] == '2/':
                ax1.set_xlim([1, 5])
                ax5.set_xlim([1, 5])
                ax1.set_xticks([1, 2, 3, 4, 5])
                ax5.set_xticks([1, 2, 3, 4, 5])
            ax1.set_ylim([-0.5, 5.5])
            ax5.set_ylim([-0.5, 5.5])
            ax1.set_yticks([0, 1, 2, 3, 4, 5])
            ax5.set_yticks([0, 1, 2, 3, 4, 5])
            ax1.legend()
            ax5.legend()
            ax1.set_xlabel('# Trades')
            ax5.set_xlabel('# Trades')
            ax1.set_ylabel('# Datapoints Traded')
            ax5.set_ylabel('# Datapoints Traded')

            # Show best/worst case beta
            for goodness in ['Best', 'Worst']:
                for instance in [1, 10, 25]:
                    with open(direc + seq + '/' + goodness.lower() + str(instance) + '.txt', 'r') as f:
                        lines = f.readlines()
                        dist = lines[0].split()[1]
                        p_true = lines[2].split()[1]
                        params = lines[7].split()[2:]
                        params = [float(p) for p in params]
                        fig, ax = plt.subplots(dpi=200)
                        x, y = beta_plot(params)
                        ax.plot(x, y)
                        ax.plot(np.repeat(float(p_true), 100), np.linspace(0, np.max(y), 100), '--k')
                        ax.legend(['Beta Distribution','$p_{true}$'])
                        ax.set_xlabel('$p$')
                        ax.set_ylabel('Probability Density')
                        ax.set_xlim([0, 1])
                        rounded = str(round(float(dist), 3))
                        if len(rounded) < 4:
                            rounded += '0'
                        ax.set_title(goodness + ' Case: Distance = ' + rounded)
                        fig.savefig(direc + seq.lower() + '/' + goodness + str(instance) + '.png')

        fig_first.tight_layout()
        fig_last.tight_layout()
        fig_first.savefig(direc + 'first_points.png')
        fig_last.savefig(direc + 'final_points.png')


def make_worst_evolve_plot(direc, goodness):
    """
    Posterior evolution plots for best/worst case markets

    goodness: either 'best' or 'worst'
    """
    fig, ((ax0, ax1), (ax5, ax_end)) = plt.subplots(nrows=2, ncols=2, dpi=200, figsize=(10, 5))
    for ax in (ax0, ax1, ax5, ax_end):
        ax.set_xlabel('p')
        ax.set_ylabel('Probability Density')
    ax0.set_title('Initialization')
    ax1.set_title('After 1 Trade Per Trader')
    ax1.set_title('After 1 Trade Per Trader')
    ax5.set_title('After 5 Trades Per Trader')
    ax_end.set_title('At Convergence')

    with open(direc + 'low_first' + '/'+goodness+'25.txt', 'r') as f:
        lines = f.readlines()
        p_true = lines[2].split()[1]
        lines = lines[10:]
        params = []
        for line in lines[::10]:
            params.append([float(num) for num in line.split()])
        x0, y0 = beta_plot(params[0])
        ax0.plot(x0, y0)
        ax0.plot(np.repeat(float(p_true), 100), np.linspace(0, 30, 100), '--k')
        x1, y1 = beta_plot(params[1])
        ax1.plot(x1, y1)
        ax1.plot(np.repeat(float(p_true), 100), np.linspace(0, 30, 100), '--k')
        x5, y5 = beta_plot(params[5])
        ax5.plot(x5, y5)
        ax5.plot(np.repeat(float(p_true), 100), np.linspace(0, 30, 100), '--k')
        xend, yend = beta_plot(params[-1])
        ax_end.plot(xend, yend)
        ax_end.plot(np.repeat(float(p_true), 100), np.linspace(0, 30, 100), '--k')

    ax0.legend(['Beta Distribution', '$p_{true}$'])
    ax1.legend(['Beta Distribution', '$p_{true}$'])
    ax5.legend(['Beta Distribution', '$p_{true}$'])
    ax_end.legend(['Beta Distribution', '$p_{true}$'])

    fig.tight_layout()
    fig.savefig(direc + goodness + '25_evolve.png')



if __name__ == '__main__':
    # make_info_plots('1round_info_sims/')
    # make_n_plots('low_n_info_sims/', np.arange(1, 26))
    # make_many_budget_plots()
    # make_worst_evolve_plot('budget_10/', 'worst')
    # make_worst_evolve_plot('budget_10/', 'best')
    # make_worst_evolve_plot('budget_simulations/')
