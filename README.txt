This repository contains classes, functions, and scripts that implement a Beta-Bernoulli market
and output results averaged over many simulations. Furthermore, we run the market with different trader
sequences, informativeness, and budgets to study the interplay between these three trader characteristics
and their eventual compensation in the market. More in-depth exposition can be found on our conference paper.
Below is an overview of each file, while a more detailed summary of each class and function can be found
within the files.

Blake Martin    WN2021



#### budget_info/ File structure ####

    beta_bernoulli.py

        DESCRIPTION
        - Contains classes and functions that model a Beta-Bernoulli market and
          agents that trade in this market.
        

    informativeness_vary.py

        DESCRIPTION
        - A script that uses the beta_bernoulli.py implementations to simulate
          many markets to convergence and average their results
        - This script assumes unlimited budget traders, but looks at the effects
          of informativeness (sample size) and sequence on compensation
        - Requires info_simulations/ folder to be located in same directory
        
        OUTPUTS
        - summary.txt:      Most relevant information is found here. Contains information of
                            the market parameters and convergence criteria as well as confidence
                            intervals for various relevant quantities like trader compensation
        - compensation.txt: Compensation of each trader averaged over n simulations.
        - params.txt:       Average alpha and beta posterior parameters after each trade


    budget_vary.py

        DESCRIPTION
        - A script that uses the beta_bernoulli.py implementations to simulate
          many markets to convergence and average their results
        - This script assumes all traders have equal informativeness (sample size),
          but looks at the effects of budget and sequence on compensation
        - Requires budget_simulations/ folder to be located in same directory
        
        OUTPUTS
        - summary.txt:  Most relevant information is found here. Contains information of
                        the market parameters and convergence criteria as well as confidence
                        intervals for various relevant quantities like budget evolution and
                        compensation
        - bestk.txt:    information about the market at round k that converged to shares
                        with the minimal L2 distance between converged prices and
                        expected prices implied by p_true
        - worstk.txt:   information about the market at round k that converged to shares
                        with the maximal L2 distance between converged prices and
                        expected prices implied by p_true
        - pointsk.txt:  Average number of points used at each trade by LB traders in
                        market round k
        

    make_plots.py

        DESCRIPTION
        - Functions for creating the various plots of different information in our
          Beta-Bernoulli market

        OUTPUTS
        - Various plots, depends on which functions are called


    randomness_sources.py

        DESCRIPTION
        - For this script, we were looking at averaging over two sources of
          randomness: data point sampling and sequence
        - To get a grid of compensation information for each sampled trader sequence
          paired with each sampled data point sequence, some structural changes needed
          to be made, which are implemented in this file
        - Requires randomness_simulations/ folder to be located in same directory

        OUTPUTS
        - high_comp.txt:  avg HI compensation at each grid location
        - low_comp.txt:   avg LI compensation at each grid location
        - points.txt:     data point sequences (rows of the compensation grids)
        - sequences.txt:  trader sequences (columns of the compensation grids)