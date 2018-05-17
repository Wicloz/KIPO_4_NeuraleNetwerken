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
    inputs = [(inputsDefault, hiddens, epochsDefault, binary, activationDefault, '2') for hiddens in hiddensTest for binary in binaryTypes]
    results = [[0 for y in hiddensTest] for x in binaryTypes]

    for output in tqdm(pool.imap_unordered(runAnalysis, inputs), total=len(inputs)):
        results[binaryTypes.index(output['binary'])][hiddensTest.index(output['hiddens'])] = output['results'][-1]

    df = pd.DataFrame(np.transpose(results), index=hiddensTest, columns=binaryTypes)
    print(df)

    plot = df.plot.line(logy=True)
    plot.set_ylabel('Mean Squared Error')
    plot.set_xlabel('Amount of Hidden Nodes')
    plt.show()

    ###############
    # Test Epochs #
    ###############
    inputs = [(inputsDefault, hiddensDefault, epochs, binary, activationDefault, '2') for epochs in epochsTest for binary in binaryTypes]
    results = [[0 for y in epochsTest] for x in binaryTypes]

    for output in tqdm(pool.imap_unordered(runAnalysis, inputs), total=len(inputs)):
        results[binaryTypes.index(output['binary'])][epochsTest.index(output['epochs'])] = output['results'][-1]

    df = pd.DataFrame(np.transpose(results), index=epochsTest, columns=binaryTypes)
    print(df)

    plot = df.plot.line()
    plot.set_ylabel('Mean Squared Error')
    plot.set_xlabel('Amount of Epochs (as input)')
    plt.show()

    ###################
    # Test Activation #
    ###################
    for binary in binaryTypes:
        inputs = [(inputsDefault, hiddensDefault, epochsDefault, binary, activation, '3') for activation in activationTypes]
        results = [[] for x in activationTypes]

        for output in tqdm(pool.imap_unordered(runAnalysis, inputs), total=len(inputs)):
            results[activationTypes.index(output['activation'])] = output['results']

        df = pd.DataFrame(np.transpose(results), index=range(1, int(epochsDefault) + 1), columns=activationTypes)
        print(df)

        plot = df.plot.line(logx=True)
        plot.set_ylabel('Mean Squared Error')
        plot.set_xlabel('Current Epoch')
        plt.show()
