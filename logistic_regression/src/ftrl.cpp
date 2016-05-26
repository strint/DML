#include "ftrl.h"

FTRL::FTRL(Load_Data* ld, int total_num_proc, int my_rank) 
    : data(ld), num_proc(total_num_proc), rank(my_rank){
    init();
}
FTRL::~FTRL(){}

void FTRL::init(){
    loc_w = new float[data->glo_fea_dim]();
    glo_w = new float[data->glo_fea_dim]();
    
    loc_g = new float[data->glo_fea_dim]();
    glo_g = new float[data->glo_fea_dim]();

    loc_z = new float[data->glo_fea_dim]();
    loc_sigma = new float[data->glo_fea_dim]();
    loc_n = new float[data->glo_fea_dim]();
    
    alpha = 1.0;
    beta = 1.0;
    lambda1 = 1.0;
    lambda2 = 1.0;
    
    step = 0;
    batch_size = 10;
}

float FTRL::sigmoid(float x){
    if(x < -30){
        return 1e-6;
    }
    else if(x > 30){
        return 1.0;
    }
    else{
        double ex = pow(2.718281828, x);
        return ex / (1.0 + ex);
    }
}

void FTRL::ftrl(){
    for(int i = 0; i < step; i++){
	for(int j = i * batch_size; j < (i + 1) * batch_size; j++){
	    float wx = 0.0;
	    for(int k = 0; k < data->fea_matrix[j].size(); j++){
		int index = data->fea_matrix[j][k].idx;
	        float val = data->fea_matrix[j][k].val;
		if(abs(loc_z[index]) <= lambda1) loc_w[index] = 0.0;
		else{
			float tmpr= 0.0;
	     		if(loc_z[index] > 0) tmpr = loc_z[index] - lambda1;
			else tmpr = loc_z[index] + lambda1;
			float tmpl = ((beta + sqrt(loc_n[index])) / alpha  + lambda2);
			loc_w[index] = tmpr / tmpl;
	    	}
		wx += loc_w[index] * val;
	    }
	    float p = 0.0;
	    p = sigmoid(wx); 

	    for(int k = 0; k < data->fea_matrix[j].size(); j++){
		int index = data->fea_matrix[j][k].idx;
                float val = data->fea_matrix[j][k].val;
		loc_g[index] = (p - data->label[j]) * val;
		loc_sigma[index] = (sqrt(loc_n[index] + loc_g[index] * loc_g[index]) - sqrt(loc_n[index])) / alpha;
		loc_z[index] += loc_g[index] - loc_sigma[index] * loc_w[index];
		loc_n[index] += loc_g[index] * loc_g[index];
	    }
	}
    }
}

void FTRL::run(){
    ftrl();
}
