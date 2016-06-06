#include "ftrl.h"

FTRL::FTRL(Load_Data* ld, int total_num_proc, int my_rank) 
    : data(ld), num_proc(total_num_proc), rank(my_rank){
    init();
}
FTRL::~FTRL(){}

void FTRL::init(){

    loc_f_val = new float[data->glo_fea_dim]();

    glo_w = new float[data->glo_fea_dim]();
    loc_w = new float[data->glo_fea_dim]();

    float* loc_v=new float[data->glo_fea_dim*factor]();
    float** loc_v_arr=new float*[data->glo_fea_dim];
    for (int i = 0; i < data->glo_fea_dim; i++)
        loc_v_arr[i]=&loc_v[i*factor];

    float glo_v = new float*[data->glo_fea_dim*factor]();
    float** glo_v_arr = new float[data->glo_fea_dim];
    for(int i = 0; i < data->glo_fea_dim; i++){
	glo_v_arr = &glo_v[i*factor];
    }
//---------------------------------------------
    loc_w_g = new float[data->glo_fea_dim]();
    glo_w_g = new float[data->glo_fea_dim]();

    float* loc_v_g=new float[data->glo_fea_dim*factor]();
    float** loc_v_g_arr=new float*[data->glo_fea_dim];
    for (int i = 0; i < data->glo_fea_dim; i++)
        loc_v_g_arr[i]=&loc_v_g[i*factor];

    float glo_v_g = new float*[data->glo_fea_dim*factor]();
    float** glo_v_g_arr = new float[data->glo_fea_dim];
    for(int i = 0; i < data->glo_fea_dim; i++){
        glo_v_g_arr = &glo_v_g[i*factor];
    }
//---------------------------------------------------------
    loc_w_z = new float[data->glo_fea_dim]();
    loc_w_sigma = new float[data->glo_fea_dim]();
    loc_w_n = new float[data->glo_fea_dim]();
    
    float* loc_v_z=new float[data->glo_fea_dim*factor]();
    float** loc_v_z_arr=new float*[data->glo_fea_dim];
    for (int i = 0; i < data->glo_fea_dim; i++)
        loc_v_z_arr[i]=&loc_v_z[i*factor];

    float* loc_v_sigma = new float[data->glo_fea_dim*factor]();
    float** loc_v_sigma_arr = new float*[data->glo_fea_dim];
    for (int i = 0; i < data->glo_fea_dim; i++)
        loc_v_sigma_arr[i]=&loc_v_sigma[i*factor];
 
    float* loc_v_n=new float[data->glo_fea_dim*factor]();
    float** loc_v_n_arr=new float*[data->glo_fea_dim];
    for (int i = 0; i < data->glo_fea_dim; i++)
        loc_v_n_arr[i]=&loc_v_n[i*factor];
//-----------------------------------------------------
    factor = 2;
    alpha = 1.0;
    beta = 1.0;
    lambda1 = 0.0;
    lambda2 = 1.0;
    
    step = 100;
    batch_size = 10;
}

float FTRL::sigmoid(float x){
    if(x < -30) return 1e-6;
    else if(x > 30) return 1.0;
    else{
        double ex = pow(2.718281828, x);
        return ex / (1.0 + ex);
    }
}

void FTRL::update_other_parameter(){
    MPI_Status status;
    for(int col = 0; col < data->glo_fea_dim; col++){
        if(rank != 0){
            MPI_Send(loc_g_w, data->glo_fea_dim, MPI_FLOAT, 0, 99, MPI_COMM_WORLD);
	    MPI_Send(loc_g_v, data->algo_fea_dim*factor, MPI_FLOAT, 0, 999, MPI_COMM_WORLD);
        }
        else if(rank == 0){
            for(int f_idx = 0; f_idx < data->glo_fea_dim; f_idx++){
                glo_g_w[f_idx] = loc_g_w[f_idx];
            }
       
  	    for(int j = 0; j < data->glo_fea_dim; j++){
		for(int k = 0; k < factor; k++){
       		    glo_v_g[j][k] = loc_v_g[j][k];
		}
	    }

            for(int ranknum = 1; ranknum < num_proc; ranknum++){
                MPI_Recv(loc_g_w, data->glo_fea_dim, MPI_FLOAT, ranknum, 99, MPI_COMM_WORLD, &status);
                for(int f_idx = 0; f_idx < data->glo_fea_dim; f_idx++){
                    glo_g_w[f_idx] += loc_g_w[f_idx];
                }
		
		MPI_Recv(loc_g_v, data->algo_fea_dim*factor, MPI_FLOAT, ranknum, 999, MPI_COMM_WORLD, &status);
		for(int f_idx = 0; f_idx < data->algo_fea_dim*factor; f_idx++){
		    loc_g_v[f_idx] = loc_g_w[f_idx];
		}

            }

            loc_sigma_w[col] = (sqrt(loc_n_w[col] + glo_g_w[col] * glo_g_w[col]) - sqrt(loc_n_w[col])) / alpha;
            loc_z_w[col] += glo_g_w[col] - loc_sigma_w[col] * loc_w[col];
            loc_n_w[col] += glo_g_w[col] * glo_g_w[col];
	
	    for(int k = 0; k < factor; k++){
		int index = col*data->algo_fea_dim + k;
	 	loc_sigma_v[index] = (sqrt(loc_n_v[index] + glo_g_v[index] * glo_g_v[index]) - sqrt(loc_n_v[index])) / alpha;
	 	loc_z_v[index] += glo_g_w[index] - loc_sigma_v[index] * loc_v[index];
		loc_n_v[index] += glo_g_v[index] * glo_g_v[index];
	    }
        }
    }//end for
}

void FTRL::update_v(){
    for(int j = 0; j < data->glo_fea_dim; j++){
        for(int k = 0; k < factor; k++){
	   float tmp_z = loc_z_v[j * data->glo_fea_dim + k];
            if(abs(tmp_z) <= lambda1){
		loc_z_v[j * data->glo_fea_dim + k] = 0.0;
	    }
	    else{
		float tmpr = 0.0;
	 	if(tmp_z > 0){
		    tmpr = tmp_z - lambda1;
		}
	        else{
		    tmpr = tmp_z + lambda1;
		}
		float tmpl = -1 * ( ( beta + sqrt(loc_n_v[j * data->glo_fea_dim + k]) ) / alpha  + lambda2);
		loc_v[j * data->glo_fea_dim + k] = tmpr / tmpl;
	    }
        }
    }
}

void FTRL::update_w(){
    for(int col = 0; col < data->glo_fea_dim; col++){
        if(abs(loc_z_w[col]) <= lambda1){
            loc_w[col] = 0.0;
        }
        else{
            float tmpr= 0.0;
            if(loc_z_w[col] > 0) tmpr = loc_z_w[col] - lambda1;
            else tmpr = loc_z_w[col] + lambda1;
            float tmpl = -1 * ( ( beta + sqrt(loc_n_w[col]) ) / alpha  + lambda2);
            loc_w[col] = tmpr / tmpl;
        }
    }
}

void FTRL::ftrl(){
    MPI_Status status;
    for(int i = 0; i < step; i++){
        int row = i * batch_size, index;
        if(rank == 0){
            update_w();
            for(int rank = 1; rank < num_proc; rank++){
                MPI_Send(loc_w, data->glo_fea_dim, MPI_FLOAT, rank, 99, MPI_COMM_WORLD);
		MPI_Send(loc_v, data->glo_fea_dim*factor, MPI_FLOAT, rank, 999, MPI_COMM_WORLD);
            }
        }
        else if(rank != 0){
            MPI_Recv(glo_w, data->glo_fea_dim, MPI_FLOAT, 0, 99, MPI_COMM_WORLD, &status);
            for(int w_idx = 0; w_idx < data->glo_fea_dim; w_idx++){
                loc_w[w_idx] = glo_w[w_idx];
            }
	    
	    MPI_Recv(glo_v, data->glo_fea_dim*factor, MPI_FLOAT, rank, 999, MPI_COMM_WORLD);
	    for(int j = 0; j < data->glo_fea_dim*factor; j++){
		loc_v[j] = glo_v[j];
	    }
        }
	while( (row < (i + 1) * batch_size) && (row < data->fea_matrix.size()) ){
	    float wx = 0.0, p = 0.0, value = 0.0;
	    for(int col = 0; col < data->fea_matrix[row].size(); col++){
	  	index = data->fea_matrix[row][col].idx;
	        value = data->fea_matrix[row][col].val;
	        wx += loc_w[index] * value;
	    }

	    float vxvx = 0.0, vvxx = 0.0;
            for(int k = 0; k < factor; k++){
		for(int col = 0; col < data->fea_matrix[row].size(); col++){
		    index = data->fea_matrix[row][col].idx;
                    value = data->fea_matrix[row][col].val;
                    vxvx += loc_v[col][k] * value;
		    vvxx += loc_v[col][k]*loc_v[col][k] * value*value;
                }
	        vxvx *= vxvx;
		vxvx -= vvxx;
		wx += vxvx;	
            }
	    p = sigmoid(wx);
            loc_g[index] += (p - data->label[row]) * value;
            ++row;
        }
	for(int col = 0; col < data->glo_fea_dim; ++col){
	    loc_g[col] /= batch_size;
	}
        update_other_parameter();     
    }//end for
}

void FTRL::run(){
    ftrl();
}
