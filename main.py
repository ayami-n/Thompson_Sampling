# Thompson Sampling

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random # for randomly put data point
import seaborn as sns

def thompson():
    # import data
    dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

    # Thompson Sampling: 3 steps

    # before steps
    N = len(dataset) # (10000 rows) == max round so far
    d = len(dataset.columns)  # (10 columns)
    ads_selected = []  # lists for selected
    total_reward = 0

    # step 1
    num_of_reeward_1 = [0] * d # the num of times the ad i got reward 1
    num_of_reeward_0 = [0] * d # the num of times the ad i got reward 0

    # step 2: the average reward of ad i up to round n
    for n_round in range(N):
        ad = 0  # traverse the columns
        max_random = 0  # check the highest convergence

        for i in range(d):  # traverse in the columns and calculate random beta
            random_beta = random.betavariate(num_of_reeward_1[i]+1, num_of_reeward_0[i]+1) # random.betavariate(alpha, beta): each rows are calculated => no need to consider 1e400

            if (max_random < random_beta):  # step 3: select maximum random
                max_random = random_beta
                ad = i  # update for ads_selected

        ads_selected.append(ad)
        reward = dataset.values[n_round, ad]

        if (dataset.values[n_round, ad] == 1):  # one of them is selected,and cannot update both at the same time
            num_of_reeward_1[ad] = num_of_reeward_1[ad] + 1
        else:
            num_of_reeward_0[ad] = num_of_reeward_0[ad] + 1

        total_reward = total_reward + reward

    # visualizing data
    plt.hist(ads_selected, color='blue')
    # sns.distplot(ads_selected, color='red')
    plt.title('Selection of Ads')
    plt.xlabel('Ads')
    plt.ylabel('Rewards')
    plt.show()


if __name__ == '__main__':
    thompson()