#include <bits/stdc++.h>
#define ll long long

using namespace std;

int main(){
    double epsilon = 1e-6;
    FILE* fp = freopen("../data/multiplier/cut_delays1.csv", "w", stdout);
    ifstream file("../data/multiplier/cut_stats.csv");
    string line;
    int ctr = 0;
    while (getline(file, line)) {
        if(ctr == 0){
            ctr++;
            continue;
        }
        istringstream iss(line);
        string token;
        vector<string> tokens;
        while (getline(iss, token, ',')) {
            tokens.push_back(token);
        }
        cout << tokens[0] << "," << tokens[1] << ",";
        double delay = stod(tokens[3]);
        if(abs(delay+1) <= epsilon) cout << "-1\n";
        else cout << delay << "\n";
    }
    file.close();
    fclose(fp);
}