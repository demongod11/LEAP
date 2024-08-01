#include <bits/stdc++.h>
#define ll long long

using namespace std;

double dfs(ll node_num, vector<ll> &primary_outputs, map<ll,ll> &primary_inputs, vector<vector<ll>> &cut_childs, vector<double> &cut_delays, vector<double> &best_arrival_time, vector<ll> &best_arrival_time_flag, map<ll,vector<ll>> &node_cut_map, map<ll, vector<ll>> &top_cuts){
    if(best_arrival_time_flag[node_num] == -1){
        best_arrival_time_flag[node_num] = 1;
        priority_queue<pair<double,ll>, vector<pair<double,ll>>, greater<pair<double,ll>>> arrival_time_queue;
        for(ll i=0; i<node_cut_map[node_num].size(); i++){
            ll cut_num = node_cut_map[node_num][i];
            if(abs(cut_delays[cut_num]+1) > 1e-6){
                double arr_time = -1;
                for(ll j=0; j<cut_childs[cut_num].size(); j++){
                    if(cut_childs[cut_num][j] != -1){
                        if(primary_inputs.find(cut_childs[cut_num][j]) != primary_inputs.end()){
                            arr_time = max(arr_time,(double)0);
                        }
                        else{
                            arr_time = max(arr_time, dfs(cut_childs[cut_num][j],primary_outputs,primary_inputs,cut_childs,cut_delays,best_arrival_time,best_arrival_time_flag,node_cut_map,top_cuts));
                        }
                    }
                }
                arr_time += cut_delays[cut_num];
                arrival_time_queue.push({arr_time, cut_num});
            }
        }
        ll counter = 0;
        double avg_arr_time = 0;
        pair<double,ll> tmp_pair;
        while((counter < 10) && (!arrival_time_queue.empty())){
            tmp_pair = arrival_time_queue.top();
            top_cuts[node_num].push_back(tmp_pair.second);
            avg_arr_time += tmp_pair.first;
            counter++;
            arrival_time_queue.pop();
        }
        if(counter == 0) avg_arr_time = 999999;
        else avg_arr_time = (avg_arr_time)/(counter);
        return best_arrival_time[node_num] = avg_arr_time;
    }
    return best_arrival_time[node_num];
}

int main(){
    clock_t clk_start, clk_end;
    clk_start = clock();
    double epsilon = 1e-6;
    
    ifstream file("../data/c6288/cut_delays.csv");
    vector<double> cut_delays;
    cut_delays.push_back((double)-1);
    string line;
    double delay_0, delay_1, delay;
    while (getline(file, line)) {
        istringstream iss(line);
        ll cut_num, phase;
        double delay;
        string token;
        vector<string> tokens;
        while (getline(iss, token, ',')) {
            tokens.push_back(token);
        }
        cut_num = stoll(tokens[0]);
        phase = stoll(tokens[1]);
        delay = stod(tokens[2]);
        if(phase == 0){
            delay_0 = delay;
        }else{
            delay_1 = delay;
            if((abs(delay_0+1) < epsilon) && (abs(delay_1+1) < epsilon)){
                cut_delays.push_back((double)-1);
            }
            else if((abs(delay_0+1) >= epsilon) && (abs(delay_1+1) < epsilon)){
                cut_delays.push_back(delay_0);
            }
            else if((abs(delay_0+1) < epsilon) && (abs(delay_1+1) >= epsilon)){
                cut_delays.push_back(delay_1);
            }
            else{
                cut_delays.push_back(min(delay_0,delay_1));
            }
        }
    }
    file.close();

    ll lc = 0;
    string str = "../data/c6288/cuts.csv";
    ifstream file2(str);
    while(getline(file2,line)){
        lc++;
    }
    file2.close();
    vector<vector<ll>> cut_childs;
    cut_childs.push_back({-1,-1,-1,-1,-1});
    FILE* fp = freopen("../data/c6288/cuts.csv", "r", stdin);
    string s,tmp;
    ll ind;
    int fl=0;
    cin>>s;
    map<ll,vector<ll>> node_cut_map;
    for(ll i=0; i<lc-1; i++){
        cin>>s;
        fl=0;tmp="";
        vector<ll> tmp_vec;
        for(ll j=0; j<s.size(); j++){
            if(s[j] == ','){
                if(fl == 0){
                    ind = stoll(tmp);
                    tmp = "";
                    fl = 1;
                }
                else if(fl == 1){
                    node_cut_map[ind].push_back(stoll(tmp));
                    tmp = "";
                    fl++;
                }
                else if(fl == 6){
                    tmp_vec.push_back(stoll(tmp));
                    break;
                }else{
                    tmp_vec.push_back(stoll(tmp));
                    tmp = "";
                    fl++;
                }
            }
            else tmp+=s[j];
        }
        cut_childs.push_back(tmp_vec);
    }
    fclose(fp);

    lc = 0;
    vector<ll> primary_outputs;
    map<ll,ll> primary_inputs;
    ifstream file3("../data/c6288/nodes.csv");
    int ctr = 0;
    while (getline(file3, line)) {
        lc++;
        if(ctr == 0){
            ctr++;
            continue;
        }
        istringstream iss2(line);
        string token;
        vector<string> tokens;
        while (getline(iss2, token, ',')) {
            tokens.push_back(token);
        }
        if(tokens[6] == "-1" && tokens[7] == "-1") primary_inputs[stoll(tokens[0])] = 1;
        else if (tokens[1] == "2") primary_outputs.push_back(stoll(tokens[0]));
    }
    file3.close();

    ll num_nodes = lc-1;
    // cout << "Num nodes are " << num_nodes << endl;
    vector<double> best_arrival_time(num_nodes,-1);
    vector<ll> best_arrival_time_flag(num_nodes,-1);
    map<ll,vector<ll>> top_cuts;
    for(ll i=0; i<primary_outputs.size(); i++){
        if(best_arrival_time_flag[primary_outputs[i]] == -1){
            dfs(primary_outputs[i], primary_outputs, primary_inputs, cut_childs, cut_delays, best_arrival_time, best_arrival_time_flag, node_cut_map, top_cuts);
        }
    }
    cout << "DFS completed successfully\n";

    ll total_final_cuts = 0;
    vector<ll> non_node_nums;    // List of all node nums for which all cuts are non-implementable according to the ML model
    fp = freopen("../data/c6288/final_cuts_10.csv", "w+", stdout);
    for(ll i=0; i<num_nodes; i++){
        if(primary_inputs.find(i) != primary_inputs.end()) continue;
        // cout << i << "\n";
        if(top_cuts.find(i) == top_cuts.end()){
            cout << i << "," << node_cut_map[i][0] << "\n";
            total_final_cuts++;
            non_node_nums.push_back(i);
        }
        else{
            if(*min_element(top_cuts[i].begin(), top_cuts[i].end()) != node_cut_map[i][0]){
                cout << i << "," << node_cut_map[i][0] << ",";
                total_final_cuts++;
            }
            else{
                cout << i << ",";
            }
            for(ll j=0; j<top_cuts[i].size()-1; j++){
                cout << top_cuts[i][j] << ",";
                total_final_cuts++;
            }
            cout << top_cuts[i][top_cuts[i].size()-1] << "\n";
            total_final_cuts++;   
        }
    }
    fclose(fp);

    fp = freopen("/dev/tty", "w", stdout);
    cout << "Number of final cuts = " << total_final_cuts << "\n";
    cout << "List of non-implementable node nums are - ";
    for(ll i=0; i<non_node_nums.size(); i++) cout << non_node_nums[i] << " ";
    cout << "\n";
    clk_end = clock() - clk_start;
    printf( "Time taken for inference = %6.4f min\n",     (float)(clk_end)/((60)*(float)(CLOCKS_PER_SEC)) );
}