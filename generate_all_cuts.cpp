#include <bits/stdc++.h>
#define ll long long

using namespace std;

int main(){
    string str = "data/b14/cuts.csv";
    ifstream file(str);
    ll lc = 0;
    string line;
    while(getline(file,line)){
        lc++;
    }
    file.close();
    FILE* fp = freopen("data/b14/cuts.csv", "r", stdin);
    string s,tmp;
    ll ind,val;
    int fl=0;
    cin>>s;
    map<ll,vector<ll>> mp;
    for(ll i=0; i<lc; i++){
        cin>>s;
        fl=0;tmp="";
        for(ll j=0; j<s.size(); j++){
            if(s[j] == ','){
                if(fl == 0){
                    ind = stoll(tmp);
                    tmp = "";
                    fl = 1;
                }
                else break;
            }
            else tmp+=s[j];
        }
        val = stoll(tmp);
        mp[ind].push_back(val);
    }
    fclose(fp);
    fp = freopen("data/b14/all_cuts.csv", "w", stdout);
    for(auto ip = mp.begin(); ip!=mp.end(); ip++){
        cout << ip->first << ",";
        for(ll i=0; i<ip->second.size()-1; i++) cout << ip->second[i] << ",";
        cout << ip->second[ip->second.size()-1] << "\n";
    }
    fclose(fp);
}
