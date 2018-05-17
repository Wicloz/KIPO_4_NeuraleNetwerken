import math
import subprocess as sp
import random
import pandas as pd
import matplotlib.pyplot as plt
from multiprocessing import Pool
from tqdm import tqdm
import numpy as np

# constant random seed
random.seed(42)

# parameters
binaryDefault = 'xor'
activationDefault = 'sigmoid'
inputsDefault = '2'
hiddensDefault = '2'
epochsDefault = '999999'

binaryTypes = ['and', 'or', 'xor']
activationTypes = ['sigmoid', 'ReLU']
hiddensTest = [str(x) for x in range(1, 10)]
epochsTest = [str(x) for x in np.logspace(1, 7)]

fileLocation = '../cmake-build-debug/KIPO_4_NeuraleNetwerken'
threadCount = 4

# functions
def runAnalysis(args):
    output = sp.check_output([fileLocation, args[0], args[1], args[2], args[3], args[4], args[5]], universal_newlines=True)
    return {
        'inputs': args[0],
        'hiddens': args[1],
        'epochs': args[2],
        'binary': args[3],
        'activation': args[4],
        'output': args[5],
        'results': [float(line.split(' ')[-1]) for line in output.splitlines()],
    }

if __name__ == '__main__':
    pool = Pool(threadCount)

    ################
    # Test Hiddens #
    # ##############
    # inputs = [(inputsDefault, hiddens, epochsDefault, binary, activationDefault, '2') for hiddens in hiddensTest for binary in binaryTypes]
    # results = [[0 for y in hiddensTest] for x in binaryTypes]
    #
    # for output in tqdm(pool.imap_unordered(runAnalysis, inputs), total=len(inputs)):
    #     results[binaryTypes.index(output['binary'])][hiddensTest.index(output['hiddens'])] = output['results'][-1]
    #
    # df = pd.DataFrame(np.transpose(results), index=hiddensTest, columns=binaryTypes)
    # print(df)
    #
    # plot = df.plot.line(logy=True)
    # plt.show()

    ###############
    # Test Epochs #
    ###############
    # inputs = [(inputsDefault, hiddensDefault, epochs, binary, activationDefault, '2') for epochs in epochsTest for binary in binaryTypes]
    # results = [[0 for y in epochsTest] for x in binaryTypes]
    #
    # for output in tqdm(pool.imap_unordered(runAnalysis, inputs), total=len(inputs)):
    #     results[binaryTypes.index(output['binary'])][epochsTest.index(output['epochs'])] = output['results'][-1]
    #
    # df = pd.DataFrame(np.transpose(results), index=epochsTest, columns=binaryTypes)
    # print(df)
    #
    # plot = df.plot.line(logx=True)
    # plt.show()

    ###################
    # Test Activation #
    ###################
    inputs = [(inputsDefault, hiddensDefault, epochsDefault, 'or', activation, '3') for activation in activationTypes]
    results = [[] for x in activationTypes]

    for output in tqdm(pool.imap_unordered(runAnalysis, inputs), total=len(inputs)):
        results[activationTypes.index(output['activation'])] = output['results']

    df = pd.DataFrame(np.transpose(results), index=range(1, int(epochsDefault) + 1), columns=activationTypes)
    print(df)

    plot = df.plot.line(logx=True)
    plt.show()

    # # make inputs
    # inputs = [(inputsDefault, hiddens, epochsDefault, binary, activationDefault, 'hiddens') for hiddens in hiddensTest for binary in binaryTypes] \
    #        + [(inputsDefault, hiddensDefault, epochs, binary, activationDefault, 'epochs') for epochs in epochsTest for binary in binaryTypes] \
    #        + [(inputsDefault, hiddensDefault, epochsDefault, binary, activation, 'activation-' + binary) for activation in activationTypes for binary in binaryTypes]
    #
    # # results storage
    # results = {
    #     'hiddens': [[] for x in binaryTypes],
    #     'epochs': [[] for x in binaryTypes],
    #     'activation-or': [[] for x in activationTypes],
    #     'activation-and': [[] for x in activationTypes],
    #     'activation-xor': [[] for x in activationTypes],
    # }
    #
    # # run
    # for output in tqdm(pool.imap_unordered(runAnalysis, inputs), total=len(inputs)):
    #     if 'activation' in output['fromTest']:
    #         results[output['fromTest']][activationTypes.index(output['activation'])] = output['results']
    #     else:
    #         results[output['fromTest']][binaryTypes.index(output['binary'])].append(output['results'][-1])
    #
    # # display results
    # dfs = {
    #     'hiddens': pd.DataFrame(results['hiddens'], index=binaryTypes, columns=hiddensTest),
    #     'epochs': pd.DataFrame(results['epochs'], index=binaryTypes, columns=epochsTest),
    #     'activation-or': pd.DataFrame(results['activation'], index=activationTypes),
    #     'activation-and': pd.DataFrame(results['activation'], index=activationTypes),
    #     'activation-xor': pd.DataFrame(results['activation'], index=activationTypes),
    # }
    # print(dfs)
    #
    # plot = dfs['hiddens'].plot.scatter

    # plt.pyplot.show()
    # plot.get_figure().savefig('results.svg', format='svg')
    # plot.get_figure().savefig('results.pdf', format='pdf')
