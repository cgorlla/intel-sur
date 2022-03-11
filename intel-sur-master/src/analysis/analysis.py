'''
Contains functions to compute and plot conditional probabilities from a dataframe
'''

# +
from collections import defaultdict

def get_totals(df):
    '''
    Gets total numbers of apps
    '''
    
    sequence = df[df['ID_INPUT'] == 3]['VALUE'].values
    pairs = defaultdict(int)

    for i in range(len(sequence)-1):
        pairs[(sequence[i], sequence[i+1])] += 1

    return pairs

def get_cond_probs(df):
    '''
    Computes conditional probabilities
    '''
    
    cond_probs = defaultdict(list)
    foregrounds = df[df['ID_INPUT'] == 3]['VALUE'].value_counts()
    pairs = sorted(get_totals(df).items(), key=lambda x: (x[0][0], -x[1]))
    # pairs = get_app_totals(df)

    for (pair, freq) in pairs:
        cond_probs[pair[0]].append((pair[1], round(freq/foregrounds[pair[0]], 3)))

    return cond_probs

def get_cond_prob_plots(df):
    '''
    Plots conditional probabilites
    '''

    cond_probs = get_cond_probs(df)
    for app in cond_probs:
        series = pd.DataFrame(cond_probs[app])
        plt.figure(figsize=(8, 6))
        plt.bar(x=series[0], height=series[1])
        plt.title(app);
        plt.xticks(rotation=90)
        plt.grid(True, alpha=0.5)
        plt.show()
