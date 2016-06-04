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
    lambda1 = 0.0;
    lambda2 = 1.0;
    
    step = 100;
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
        int row = i * batch_size;
	while( (row < (i + 1) * batch_size) && (row < data->fea_matrix.size()) ){
	    float wx = 0.0;
	    for(int col = 0; col < data->fea_matrix[row].size(); col++){
		int index = data->fea_matrix[row][col].idx;
	        float val = data->fea_matrix[row][col].val;
		if(abs(loc_z[index]) <= lambda1){
	            loc_w[index] = 0.0;
		}
		else{
		    float tmpr= 0.0;
	     	    if(loc_z[index] > 0) tmpr = loc_z[index] - lambda1;
		    else tmpr = loc_z[index] + lambda1;
		    float tmpl = -1 * ( ( beta + sqrt(loc_n[index]) ) / alpha  + lambda2);
		    loc_w[index] = tmpr / tmpl;
	    	}
		wx += loc_w[index] * val;
	    }
	    float p = 0.0;
	    p = sigmoid(wx); 
	    MPI_Status status;
	    for(int col = 0; col < data->fea_matrix[row].size(); col++){
		int index = data->fea_matrix[row][col].idx;
                float value = data->fea_matrix[row][col].val;
		loc_g[index] = (p - data->label[row]) * value;

		if(rank != 0){
		    MPI_Send(loc_g, data->glo_fea_dim, MPI_FLOAT, 0, 99, MPI_COMM_WORLD);
		}
		else if(rank == 0){
		    for(int f_idx = 0; f_idx < data->glo_fea_dim; f_idx++){
			glo_g[f_idx] = loc_g[f_idx];
		    }
		    for(int ranknum = 1; ranknum < num_proc; ranknum++){
		        MPI_Recv(loc_g, data->glo_fea_dim, MPI_FLOAT, ranknum, 99, MPI_COMM_WORLD, &status);
			for(int f_idx = 0; f_idx < data->glo_fea_dim; f_idx++){
                            glo_g[f_idx] += loc_g[f_idx];
                        }
		    }
		}

		loc_sigma[index] = (sqrt(loc_n[index] + glo_g[index] * glo_g[index]) - sqrt(loc_n[index])) / alpha;
		loc_z[index] += glo_g[index] - loc_sigma[index] * loc_w[index];
		loc_n[index] += glo_g[index] * glo_g[index];
	    }

	    if(rank == 0){
	        for(int jj = 0; jj < data->glo_fea_dim; jj++){
		    std::cout<<loc_w[jj]<<" ";
	        }
            }
	}
    }
}

void FTRL::run(){
    ftrl();
    std::cout<<"train end~"<<std::endl;
}
