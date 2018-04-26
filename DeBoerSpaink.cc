//
// C++-programma voor neuraal netwerk (NN) met één output-knoop
// Zie www.liacs.leidenuniv.nl/~kosterswa/AI/nnhelp.pdf
// 19 april 2018
// Compileren: g++ -Wall -O2 -o nn nn.cc
// Gebruik:    ./nn <inputs> <hiddens> <epochs> <or|and|xor>
// Voorbeeld:  ./nn 2 3 100000
//

#include <iostream>
#include <cmath>
#include <fstream>
#include <cstdlib>

using namespace std;

const int MAX = 20;
const double ALPHA = 0.1;
const double BETA = 1.0;

double dRand(double fMin, double fMax) {
    double f = (double) random() / RAND_MAX;
    return fMin + f * (fMax - fMin);
}

double g(double x) {
    return 1 / (1 + exp(-BETA * x));
}

double gprime(double x) {
    return BETA * g(x) * (1 - g(x));
}

bool getTarget(int type, double input[MAX], int inputs) {
    switch (type) {
        case 1:
            for (int i = 1; i < inputs; ++i) {
                if (input[i]) {
                    return true;
                }
            }
            return false;

        case 2:
            for (int i = 1; i < inputs; ++i) {
                if (!input[i]) {
                    return false;
                }
            }
            return true;

        case 3:
            auto xorOut = static_cast<bool>(input[1]);
            for (int i = 2; i < inputs; ++i) {
                xorOut = xorOut != input[i];
            }
            return xorOut;
    }
}

int main(int argc, char* argv[]) {

    int inputs, hiddens;            // aantal invoer- en verborgen knopen
    double input[MAX];              // de invoer is input[1]...input[inputs]
    double inputtohidden[MAX][MAX]; // gewichten van invoerknopen 0..inputs naar verborgen knopen 1..hiddens
    double hiddentooutput[MAX];     // gewichten van verborgen knopen 0..hiddens naar de ene uitvoerknoop
    double inhidden[MAX];           // invoer voor de verborgen knopen 1..hiddens
    double acthidden[MAX];          // en de uitvoer daarvan
    double inoutput;                // invoer voor de ene uitvoerknoop
    double netoutput;               // en de uitvoer daarvan: de net-uitvoer
    double target;                  // gewenste uitvoer
    double error;                   // verschil tussen gewenste en geproduceerde uitvoer
    double delta;                   // de delta voor de uitvoerknoop
    double deltahidden[MAX];        // de delta's voor de verborgen knopen 1..hiddens
    int epochs;                     // aantal trainingsvoorbeelden
    int binary;                     // binaire functie om te leren (or=1, and=2, xor=3)

    if (argc != 5 || (string(argv[4]) != "or" && string(argv[4]) != "and" && string(argv[4]) != "xor")) {
        cout << "Gebruik: " << argv[0] << " <inputs> <hiddens> <epochs> <or|and|xor>" << endl;
        return 1;
    }

    inputs = atoi(argv[1]) + 1;
    hiddens = atoi(argv[2]) + 1;
    epochs = atoi(argv[3]);
    binary = string(argv[4]) == "or" ? 1 : binary;
    binary = string(argv[4]) == "and" ? 2 : binary;
    binary = string(argv[4]) == "xor" ? 3 : binary;
    input[0] = -1;                  // invoer bias-knoop: altijd -1
    acthidden[0] = -1;              // verborgen bias-knoop: altijd -1
    srand(time(nullptr));
    srandom(time(nullptr));

    //TODO-1 initialiseer de gewichten random tussen -1 en 1:
    // inputtohidden en hiddentooutput
    // rand ( ) levert geheel randomgetal tussen 0 en RAND_MAX; denk aan casten

    for (int i = 0; i < inputs; ++i) {
        for (int j = 0; j < hiddens; ++j) {
            inputtohidden[i][j] = dRand(-1.0, 1.0);
        }
    }

    for (int i = 0; i < hiddens; ++i) {
        hiddentooutput[i] = dRand(-1.0, 1.0);
    }

    for (int i = 0; i < epochs; ++i) {

        //TODO-2 lees een voorbeeld in naar input en target, of genereer dat ter plekke:
        // als voorbeeld: de XOR-functie, waarvoor geldt dat inputs = 2
        // int x = rand ( ) % 2; int y = rand ( ) % 2; int dexor = ( x + y ) % 2;
        // input[1] = x; input[2] = y; target = dexor;

        for (int j = 1; j < inputs; ++j) {
            input[j] = random() % 2;
        }
        target = getTarget(binary, input, inputs);

        //TODO-3 stuur het voorbeeld door het netwerk
        // reken inhidden's uit, acthidden's, inoutput en netoutput

        for (int j = 1; j < hiddens; ++j) {
            inhidden[j] = 0;
            for (int k = 0; k < inputs; ++k) {
                inhidden[j] += input[k] * inputtohidden[k][j];
            }
            acthidden[j] = g(inhidden[j]);
        }

        inoutput = 0;
        for (int j = 0; j < hiddens; ++j) {
            inoutput += acthidden[j] * hiddentooutput[j];
        }
        netoutput = g(inoutput);

        //TODO-4 bereken error, delta, en deltahidden

        error = target - netoutput;
        delta = error * gprime(inoutput);
        for (int j = 0; j < hiddens; ++j) {
            deltahidden[j] = gprime(inhidden[j]) * hiddentooutput[j] * delta;
        }

        //TODO-5 update gewichten hiddentooutput en inputtohidden

        for (int j = 0; j < hiddens; ++j) {
            hiddentooutput[j] += ALPHA * acthidden[j] * delta;
        }

        for (int j = 1; j < hiddens; ++j) {
            for (int k = 0; k < inputs; ++k) {
                inputtohidden[k][j] += ALPHA * input[k] * deltahidden[j];
            }
        }

    }

    //TODO-6 beoordeel het netwerk en rapporteer

    for (int i = 0; i < pow(2, inputs - 1); ++i) {

        for (int j = 1; j < inputs; ++j) {
            input[j] = (i >> (inputs - j - 1)) & 1;
        }
        target = getTarget(binary, input, inputs);

        for (int j = 1; j < hiddens; ++j) {
            inhidden[j] = 0;
            for (int k = 0; k < inputs; ++k) {
                inhidden[j] += input[k] * inputtohidden[k][j];
            }
            acthidden[j] = g(inhidden[j]);
        }

        inoutput = 0;
        for (int j = 0; j < hiddens; ++j) {
            inoutput += acthidden[j] * hiddentooutput[j];
        }
        netoutput = g(inoutput);

        error = target - netoutput;

        for (int j = 1; j < inputs; ++j) {
            cout << input[j] << " ";
        }
        cout << round(netoutput) << " - Error: " << error << endl;

    }

    return 0;
}
