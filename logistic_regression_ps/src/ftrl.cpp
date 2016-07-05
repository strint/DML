#include "ftrl.h"

FTRL::FTRL(Load_Data* load_data) 
    : data(load_data){
    init();
}
FTRL::~FTRL(){}

void FTRL::init(){
    alpha = 1.0;
    beta = 1.0;
    lambda1 = 0.0;
    lambda2 = 1.0;
    bias = 0.1; 
    step = 1000;
}

float FTRL::sigmoid(float x){
    if(x < -30) return 1e-6;
    else if(x > 30) return 1.0;
    else{
        double ex = pow(2.718281828, x);
        return ex / (1.0 + ex);
    }
}

void FTRL::updateW(std::vector<ps::Key> &keys, std::vector<FTRLEntry>& entrys, std::vector<float> &g){
    for(int i = 0; i < keys.size(); i++){
	float sqrt_n = entrys[i].sq_cum_grad;
	float sqrt_n_new = sqrt(sqrt_n * sqrt_n + g[i] * g[i]);
        entrys[i].z += g[i] - (sqrt_n_new - sqrt_n);
	entrys[i].sq_cum_grad = sqrt_n_new;
	float z = entrys[i].z;
        if(abs(z) <= lambda1){
            entrys[i].w = 0.0;
        }
        else{
            float tmpr= 0.0;
            if(z >= 0) tmpr = z - lambda1;
            else tmpr = z + lambda1;
            float tmpl = -1 * ( ( beta + entrys[i].sq_cum_grad - sqrt_n) / alpha  + lambda2);
            entrys[i].w = tmpr / tmpl;
        }
    }
    kv_->Wait(kv_->Push(keys, entrys));
}

void FTRL::run(){
    for(int i = 0; i < step; i++){
	data->load_data_minibatch(1000);
        std::vector<FTRLEntry> entrys;	    
        std::vector<float> g;
	for(int i = 0; i < data->fea_matrix.size(); i++){
  	    std::vector<ps::Key> keys;
	    std::vector<float> values;
	    float wx = bias;
            for(int j = 0; j < data->fea_matrix[i].size(); j++){
	        long int index = data->fea_matrix[i][j].idx;
	        keys.push_back(index);
	        float value = data->fea_matrix[i][j].val;
		values.push_back(value);
            }
	    kv_->Wait(kv_->Pull(keys, &entrys));
	    for(int j = 0; j < entrys.size(); j++){
 		wx += entrys[j].w * values[j];
 	    } 
	    float pctr = sigmoid(wx);
	    g.resize(keys.size());
	    for(int j = 0; j < keys.size(); j++){
                g[j] += (pctr - data->label[i]) * values[j];
	    }
            updateW(keys, entrys, g);     
        }//end for
    }
}
