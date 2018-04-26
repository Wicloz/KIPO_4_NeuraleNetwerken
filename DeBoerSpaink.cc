
//
// C++-programma voor neuraal netwerk (NN) met één output-knoop
// Zie www.liacs.leidenuniv.nl/~kosterswa/AI/nnhelp.pdf
// 19 april 2018
// Compileren: g++ -Wall -O2 -o nn nn.cc
// Gebruik:    ./nn <inputs> <hiddens> <epochs>
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

// g-functie (sigmoid)
double g(double x) {
    return 1 / (1 + exp(-BETA * x));
}//g

// afgeleide van g
double gprime(double x) {
    return BETA * g(x) * (1 - g(x));
}//gprime

int main(int argc, char* argv[]) {

    int inputs, hiddens;            // aantal invoer- en verborgen knopen
    double input[MAX];              // de invoer is input[1]...input[inputs]
    double inputtohidden[MAX][MAX]; // gewichten van invoerknopen 0..inputs
    // naar verborgen knopen 1..hiddens
    double hiddentooutput[MAX];     // gewichten van verborgen knopen 0..hiddens
    // naar de ene uitvoerknoop
    double inhidden[MAX];           // invoer voor de verborgen knopen 1..hiddens
    double acthidden[MAX];          // en de uitvoer daarvan
    double inoutput;                // invoer voor de ene uitvoerknoop
    double netoutput;               // en de uitvoer daarvan: de net-uitvoer
    double target;                  // gewenste uitvoer
    double error;                   // verschil tussen gewenste en
    // geproduceerde uitvoer
    double delta;                   // de delta voor de uitvoerknoop
    double deltahidden[MAX];        // de delta's voor de verborgen
    // knopen 1..hiddens
    int epochs;                     // aantal trainingsvoorbeelden
    int i, j, k;                    // tellertjes
    //int seed = 1234;                // eventueel voor random-generator

    if (argc != 4) {
        cout << "Gebruik: " << argv[0] << " <inputs> <hiddens> <epochs>" << endl;
        return 1;
    }//if
    inputs = atoi(argv[1]);
    hiddens = atoi(argv[2]);
    epochs = atoi(argv[3]);
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

    for (i = 0; i < epochs; i++) {

        //TODO-2 lees een voorbeeld in naar input en target, of genereer dat ter plekke:
        // als voorbeeld: de XOR-functie, waarvoor geldt dat inputs = 2
        // int x = rand ( ) % 2; int y = rand ( ) % 2; int dexor = ( x + y ) % 2;
        // input[1] = x; input[2] = y; target = dexor;

        target = true;
        for (int j = 0; j < inputs; ++j) {
            input[j] = random() % 2;
            if (!input[j]) {
                target = false;
            }
        }

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

        //TODO-5 update gewichten hiddentooutput en inputtohidden

    }//for

    //TODO-6 beoordeel het netwerk en rapporteer

    return 0;
}//main

