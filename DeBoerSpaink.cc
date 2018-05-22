/* DeBoerSpaink.cc
 * Wilco de Boer, Hermes Spaink
 * Heeft minstens c++11 nodig voor compilatie
 * Heeft veel command line argumenten nodig,
 * zie output en commentaar bovenaan main
 */

#include <iostream>
#include <cmath>
using namespace std;

const int MAX = 20;
double ALPHA = 0.2;
double BETA = 1.0;

inline double randomDouble(const double& min, const double& max) {
    double d = (double) rand() / RAND_MAX;
    return min + d * (max - min);
}

inline bool randomBool() {
    return static_cast<bool>(rand() % 2);
}

inline double g(const double& x) {
    return 1 / (1 + exp(-BETA * x));
}

inline double gPrime(const double& x) {
    return BETA * g(x) * (1 - g(x));
}

inline double relu(const double& x) {
    return max(0.0, x);
}

inline double reluPrime(const double& x) {
    return x > 0.0 ? 1.0 : 0.0;
}

inline bool getTarget(const int& type, const double(& input)[MAX], const int& inputs) {
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
    double totaalError;             // totaal van het kwadraat van de errors
    bool useRelu;                   // of de relu of sigmoid als activatie functie wordt gebruikt
    int outputMode;                 // hoe de output weergeven wordt (1=laatste+tabel, 2=laatste, 3=alle)

    if (argc != 8 || (string(argv[4]) != "or" && string(argv[4]) != "and" && string(argv[4]) != "xor") || (string(argv[5]) != "sigmoid" && string(argv[5]) != "ReLU")) {
        cout << "Gebruik: " << argv[0] << " <inputs> <hiddens> <epochs> <or|and|xor> <sigmoid|ReLU> <output> <alpha>" << endl;
        return 1;
    }

    inputs = stoi(argv[1]) + 1;
    hiddens = stoi(argv[2]) + 1;
    epochs = stoi(argv[3]);
    binary = string(argv[4]) == "or" ? 1 : 0;
    binary = string(argv[4]) == "and" ? 2 : binary;
    binary = string(argv[4]) == "xor" ? 3 : binary;
    useRelu = string(argv[5]) == "ReLU";
    outputMode = stoi(argv[6]);
    ALPHA = stod(argv[7]);

    input[0] = -1;
    acthidden[0] = -1;
    srand(42);

    //TODO-1 initialiseer de gewichten random tussen -1 en 1:
    // inputtohidden en hiddentooutput
    // rand ( ) levert geheel randomgetal tussen 0 en RAND_MAX; denk aan casten

    for (int j = 1; j < hiddens; ++j) {
        for (int k = 0; k < inputs; ++k) {
            inputtohidden[k][j] = randomDouble(-1.0, 1.0);
        }
    }

    for (int j = 0; j < hiddens; ++j) {
        hiddentooutput[j] = randomDouble(-1.0, 1.0);
    }

    for (int i = 0; i < epochs; ++i) {
        //TODO-6 beoordeel het netwerk en rapporteer

        if (outputMode == 3) {
            totaalError = 0;
            for (int z = 0; z < pow(2, inputs - 1); ++z) { // rij in truth table
                for (int j = 1; j < inputs; ++j) { // kolom in truth table
                    input[j] = (z >> (inputs - j - 1)) & 1; // nummer (1 of 0) dat daar hoort
                }
                target = getTarget(binary, input, inputs);

                for (int j = 1; j < hiddens; ++j) {
                    inhidden[j] = 0;
                    for (int k = 0; k < inputs; ++k) {
                        inhidden[j] += input[k] * inputtohidden[k][j];
                    }
                    acthidden[j] = useRelu ? relu(inhidden[j]) : g(inhidden[j]);
                }

                inoutput = 0;
                for (int j = 0; j < hiddens; ++j) {
                    inoutput += acthidden[j] * hiddentooutput[j];
                }
                netoutput = useRelu ? relu(inoutput) : g(inoutput);

                error = target - netoutput;
                totaalError += pow(error, 2);
            }

            cout << "Mean squared error: " << totaalError / pow(2, inputs - 1) << endl;
        }

        //TODO-2 lees een voorbeeld in naar input en target, of genereer dat ter plekke:
        // als voorbeeld: de XOR-functie, waarvoor geldt dat inputs = 2
        // int x = rand ( ) % 2; int y = rand ( ) % 2; int dexor = ( x + y ) % 2;
        // input[1] = x; input[2] = y; target = dexor;

        target = randomBool();
        do {
            for (int j = 1; j < inputs; ++j) {
                switch (binary) {
                    case 1:
                        if (target == 0) {
                            input[j] = 0;
                        } else {
                            input[j] = randomBool();
                        }
                        break;
                    case 2:
                        if (target == 1) {
                            input[j] = 1;
                        } else {
                            input[j] = randomBool();
                        }
                        break;
                    case 3:
                        input[j] = randomBool();
                        break;
                }
            }
        } while (target != getTarget(binary, input, inputs));

        //TODO-3 stuur het voorbeeld door het netwerk
        // reken inhidden's uit, acthidden's, inoutput en netoutput

        for (int j = 1; j < hiddens; ++j) {
            inhidden[j] = 0;
            for (int k = 0; k < inputs; ++k) {
                inhidden[j] += input[k] * inputtohidden[k][j];
            }
            acthidden[j] = useRelu ? relu(inhidden[j]) : g(inhidden[j]);
        }

        inoutput = 0;
        for (int j = 0; j < hiddens; ++j) {
            inoutput += acthidden[j] * hiddentooutput[j];
        }
        netoutput = useRelu ? relu(inoutput) : g(inoutput);

        //TODO-4 bereken error, delta, en deltahidden

        error = target - netoutput;
        delta = error * (useRelu ? reluPrime(inoutput) : gPrime(inoutput));
        for (int j = 0; j < hiddens; ++j) {
            deltahidden[j] = (useRelu ? reluPrime(inhidden[j]) : gPrime(inhidden[j])) * hiddentooutput[j] * delta;
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

    totaalError = 0;
    for (int z = 0; z < pow(2, inputs - 1); ++z) { // rij in truth table
        for (int j = 1; j < inputs; ++j) { // kolom in truth table
            input[j] = (z >> (inputs - j - 1)) & 1; // nummer (1 of 0) dat daar hoort
        }
        target = getTarget(binary, input, inputs);

        for (int j = 1; j < hiddens; ++j) {
            inhidden[j] = 0;
            for (int k = 0; k < inputs; ++k) {
                inhidden[j] += input[k] * inputtohidden[k][j];
            }
            acthidden[j] = useRelu ? relu(inhidden[j]) : g(inhidden[j]);
        }

        inoutput = 0;
        for (int j = 0; j < hiddens; ++j) {
            inoutput += acthidden[j] * hiddentooutput[j];
        }
        netoutput = useRelu ? relu(inoutput) : g(inoutput);

        error = target - netoutput;
        totaalError += pow(error, 2);

        if (outputMode == 1) {
            for (int j = 1; j < inputs; ++j) {
                cout << input[j] << " ";
            }
            cout << "> " << round(netoutput) << " (Error: " << (error < 0 ? "" : " ") << error << ")" << endl;
        }
    }

    cout << "Mean Squared Error: " << totaalError / pow(2, inputs - 1) << endl;

    return 0;
}
