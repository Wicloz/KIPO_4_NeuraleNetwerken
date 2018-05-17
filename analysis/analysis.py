import subprocess as sp
import random
import pandas as pd
import matplotlib.pyplot as plt
from multiprocessing import Pool
from tqdm import tqdm
import numpy as np
from time import time

begin = time()

# constant random seed
random.seed(42)

# parameters
binaryDefault = 'xor'
activationDefault = 'sigmoid'
inputsDefault = '2'
hiddensDefault = '2'
epochsDefault = '99999'
alphaDefault = '0.1'

binaryTypes = ['and', 'or', 'xor']
activationTypes = ['sigmoid', 'ReLU']
hiddensTest = []
for x in [str(int(round(x))) for x in np.logspace(0, 2, 20)]:
    if x not in hiddensTest:
        hiddensTest.append(x)
epochsTest = []
for x in [str(int(round(x))) for x in np.logspace(0, 7, 100)]:
    if x not in epochsTest:
        epochsTest.append(x)
alphaTest = []
for x in [str(round(x / 100, 4)) for x in np.logspace(0, 2, 40)]:
    if x not in epochsTest:
        alphaTest.append(x)

fileLocation = '../cmake-build-debug/KIPO_4_NeuraleNetwerken'
threadCount = None


# functions
def runAnalysis(args):
    output = sp.check_output([fileLocation, args[0], args[1], args[2], args[3], args[4], args[5], args[6]],
                             universal_newlines=True)
    return {
        'inputs': args[0],
        'hiddens': args[1],
        'epochs': args[2],
        'binary': args[3],
        'activation': args[4],
        'output': args[5],
        'alpha': args[6],
        'results': [float(line.split(' ')[-1]) for line in output.splitlines()],
    }


if __name__ == '__main__':
    pool = Pool(threadCount)

    ##############
    # Test Alpha #
    ##############
    inputs = [(inputsDefault, hiddensDefault, epochsDefault, binary, activationDefault, '2', alpha) for alpha in
              alphaTest for binary in binaryTypes]
    results = [[0 for y in alphaTest] for x in binaryTypes]

    for output in tqdm(pool.imap_unordered(runAnalysis, inputs), total=len(inputs)):
        results[binaryTypes.index(output['binary'])][alphaTest.index(output['alpha'])] = output['results'][-1]

    df = pd.DataFrame(np.transpose(results), index=[float(x) for x in alphaTest], columns=binaryTypes)

    plot = df.plot.line(logx=True,
                        title='Fout na trainen met verschillende leersnelheden')
    plot.set_ylabel('Gemiddelde Kwadratische Fout')
    plot.set_xlabel('Alpha')
    plt.show(logx=True)
    plt.show()

    ################
    # Test Hiddens #
    # ##############
    inputs = [(inputsDefault, hiddens, epochsDefault, binary, activationDefault, '2', alphaDefault) for hiddens in
              hiddensTest for binary in binaryTypes]
    results = [[0 for y in hiddensTest] for x in binaryTypes]

    for output in tqdm(pool.imap_unordered(runAnalysis, inputs), total=len(inputs)):
        results[binaryTypes.index(output['binary'])][hiddensTest.index(output['hiddens'])] = output['results'][-1]

    df = pd.DataFrame(np.transpose(results), index=[int(x) for x in hiddensTest], columns=binaryTypes)

    plot = df.plot.line(logx=True, title='Fout per Aantal Verborgen Knopen')
    plot.set_ylabel('Gemiddelde Kwadratische Fout')
    plot.set_xlabel('Aantal Verborgen Knopen')
    plt.show(logx=True)

    ###############
    # Test Epochs #
    ###############
    inputs = [(inputsDefault, hiddensDefault, epochs, binary, activationDefault, '2', alphaDefault) for epochs in
              epochsTest for binary in binaryTypes]
    results = [[0 for y in epochsTest] for x in binaryTypes]

    for output in tqdm(pool.imap_unordered(runAnalysis, inputs), total=len(inputs)):
        results[binaryTypes.index(output['binary'])][epochsTest.index(output['epochs'])] = output['results'][-1]

    df = pd.DataFrame(np.transpose(results), index=[int(x) for x in epochsTest], columns=binaryTypes)

    plot = df.plot.line(logx=True, title='Fout per Aantal Epochs')
    plot.set_ylabel('Gemiddelde Kwadratische Fout')
    plot.set_xlabel('Aantal Epochs (als input)')
    plt.show()

    ###################
    # Test Activation #
    ###################
    for binary in binaryTypes:
        inputs = [(inputsDefault, hiddensDefault, epochsDefault, binary, activation, '3', alphaDefault) for activation
                  in activationTypes]
        results = [[] for x in activationTypes]

        for output in tqdm(pool.imap_unordered(runAnalysis, inputs), total=len(inputs)):
            results[activationTypes.index(output['activation'])] = output['results']

        df = pd.DataFrame(np.transpose(results), index=range(1, int(epochsDefault) + 1), columns=activationTypes)

        plot = df.plot.line(logx=True, title='Fout tijdens het trainen voor de \'' + binary + '\' functie')
        plot.set_ylabel('Gemiddelde Kwadratische Fout')
        plot.set_xlabel('Huidige Epoch')
        plt.show()

print('Total time taken:', time() - begin)
