#include "ftrl.h"
#define factor 2

FTRL::FTRL(Load_Data* ld, int total_num_proc, int my_rank) 
    : data(ld), num_proc(total_num_proc), rank(my_rank){
    init();
}
FTRL::~FTRL(){}

void FTRL::init(){
    v_dim = data->glo_fea_dim*factor;
    
    temp_value = new float[data->glo_fea_dim]();
//----------------------w--------------------- 
    glo_w = new float[data->glo_fea_dim]();
    loc_w = new float[data->glo_fea_dim]();
//----------------------v---------------------
    loc_v=new float[v_dim]();
    loc_v_arr=new float*[data->glo_fea_dim];
    for (int i = 0; i < data->glo_fea_dim; i++)
        loc_v_arr[i]=&loc_v[i*factor];

    glo_v = new float[v_dim]();
    glo_v_arr = new float*[data->glo_fea_dim];
    for(int i = 0; i < data->glo_fea_dim; i++){
	glo_v_arr[i] = &glo_v[i*factor];
    }
//----------------------g_w----------------------
    loc_g_w = new float[data->glo_fea_dim]();
    glo_g_w = new float[data->glo_fea_dim]();
//----------------------g_v----------------------
    loc_g_v=new float[v_dim]();
    loc_g_v_arr=new float*[data->glo_fea_dim];
    for (int i = 0; i < data->glo_fea_dim; i++)
        loc_g_v_arr[i]=&loc_g_v[i*factor];

    glo_g_v = new float[v_dim]();
    glo_g_v_arr = new float*[data->glo_fea_dim];
    for(int i = 0; i < data->glo_fea_dim; i++){
        glo_g_v_arr[i] = &glo_g_v[i*factor];
    }
//------------------------w---------------------
    loc_z_w = new float[data->glo_fea_dim]();
    loc_sigma_w = new float[data->glo_fea_dim]();
    loc_n_w = new float[data->glo_fea_dim]();
//------------------------v---------------------    
    loc_z_v=new float[v_dim]();
    loc_z_v_arr=new float*[data->glo_fea_dim];
    for (int i = 0; i < data->glo_fea_dim; i++)
        loc_z_v_arr[i]=&loc_z_v[i*factor];

    loc_sigma_v = new float[v_dim]();
    loc_sigma_v_arr = new float*[data->glo_fea_dim];
    for (int i = 0; i < data->glo_fea_dim; i++)
        loc_sigma_v_arr[i]=&loc_sigma_v[i*factor];
 
    loc_n_v=new float[v_dim]();
    loc_n_v_arr=new float*[data->glo_fea_dim];
    for (int i = 0; i < data->glo_fea_dim; i++)
        loc_n_v_arr[i]=&loc_n_v[i*factor];
//-----------------------------------------------------
    alpha = 1.0;
    beta = 1.0;
    lambda1 = 0.0;
    lambda2 = 1.0;
    
    step = 100;
    batch_size = 10;
    bias = 1;
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
    std::cout<<"update_other_parameter"<<std::endl;
    MPI_Status status;
    if(rank != 0){
	MPI_Send(loc_g_w, data->glo_fea_dim, MPI_FLOAT, 0, 99, MPI_COMM_WORLD);
	MPI_Send(loc_g_v, data->glo_fea_dim*factor, MPI_FLOAT, 0, 999, MPI_COMM_WORLD);
    }
    else if(rank == 0){
	for(int ranknum = 1; ranknum < num_proc; ranknum++){
	    MPI_Recv(glo_g_w, data->glo_fea_dim, MPI_FLOAT, ranknum, 99, MPI_COMM_WORLD, &status);
	    for(int j = 0; j < data->glo_fea_dim; j++){
	       loc_g_w[j] += glo_g_w[j];
	    }
	    MPI_Recv(glo_g_v, data->glo_fea_dim*factor, MPI_FLOAT, ranknum, 999, MPI_COMM_WORLD, &status);
	    for(int j = 0; j < data->glo_fea_dim; j++){
		for(int k = 0; k < factor; k++){
		    loc_g_v_arr[j][k] = glo_g_v_arr[j][k];
		}
            }
	}
        for(int col = 0; col < data->glo_fea_dim; col++){
	    loc_sigma_w[col] = (sqrt(loc_n_w[col] + loc_g_w[col] * loc_g_w[col]) - sqrt(loc_n_w[col])) / alpha;
	    loc_z_w[col] += loc_g_w[col] - loc_sigma_w[col] * loc_w[col];
	    loc_n_w[col] += loc_g_w[col] * loc_g_w[col];

	    for(int k = 0; k < factor; k++){
		int index = col*data->glo_fea_dim + k;
		loc_sigma_v[index] = (sqrt(loc_n_v[index] + loc_g_v[index] * loc_g_v[index]) - sqrt(loc_n_v[index])) / alpha;
		loc_z_v[index] += loc_g_w[index] - loc_sigma_v[index] * loc_v[index];
		loc_n_v[index] += loc_g_v[index] * loc_g_v[index];
	    }
	}
    }
}

void FTRL::update_g(){
    for(int col = 0; col < data->glo_fea_dim; col++){
        loc_g_w[col] = loss_sum * temp_value[col] / batch_size;
        float vx = 0.0;
        for(int k = 0; k < factor; k++){
            for(int j = 0; j != col && j < data->glo_fea_dim; j++){
                vx += loc_v_arr[j][k] * temp_value[j] / batch_size;
            }
            vx *= temp_value[col] / batch_size;;
            loc_g_v_arr[col][k] = loss_sum * vx;
        }
    }
    for(int col = 0; col < data->glo_fea_dim; ++col){
        loc_g_w[col] /= batch_size;
    }
    for(int j = 0; j < data->glo_fea_dim; ++j){
        for(int k = 0; k < factor; k++){
            loc_g_v_arr[j][k] /= batch_size;
        }
    }
}

void FTRL::update_v(){
     for(int j = 0; j < data->glo_fea_dim; j++){
        for(int k = 0; k < factor; k++){
           float tmp_z = loc_z_v_arr[j][k];
            if(abs(tmp_z) <= lambda1){
                loc_z_v_arr[j][k] = 0.0;
            }
            else{
                float tmpr = 0.0;
                if(tmp_z >= 0){
                    tmpr = tmp_z - lambda1;
                }
                else{
                    tmpr = tmp_z + lambda1;
                }
                float tmpl = -1 * ( ( beta + sqrt(loc_n_v_arr[j][k]) ) / alpha  + lambda2);
                loc_v_arr[j][k] = tmpr / tmpl;
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
            if(loc_z_w[col] >= 0) tmpr = loc_z_w[col] - lambda1;
            else tmpr = loc_z_w[col] + lambda1;
            float tmpl = -1 * ( ( beta + sqrt(loc_n_w[col]) ) / alpha  + lambda2);
            loc_w[col] = tmpr / tmpl;
            //std::cout<<loc_w[col]<<std::endl;
        }
    }
}

void FTRL::update_parameter(){
    MPI_Status status;
    if(rank == 0){
        update_w();
        update_v();
        for(int r = 1; r < num_proc; r++){
            std::cout<<"master node send loc_w and loc_v to worker node "<<r<<std::endl;
            MPI_Send(loc_w, data->glo_fea_dim, MPI_FLOAT, r, 99, MPI_COMM_WORLD);
            MPI_Send(loc_v, data->glo_fea_dim*factor, MPI_FLOAT, r, 999, MPI_COMM_WORLD);
        }
    }
    else if(rank != 0){
	MPI_Recv(glo_w, data->glo_fea_dim, MPI_FLOAT, 0, 99, MPI_COMM_WORLD, &status);
        for(int j = 0; j < data->glo_fea_dim; j++){
            loc_w[j] = glo_w[j];
            //std::cout<<loc_w[j]<<" ";
        }
        //std::cout<<std::endl;
        MPI_Recv(glo_v, data->glo_fea_dim*factor, MPI_FLOAT, 0, 999, MPI_COMM_WORLD, &status);
        for(int j = 0; j < data->glo_fea_dim*factor; j++){
            loc_v[j] = glo_v[j];
        }
    }
}


void FTRL::ftrl(){
    MPI_Status status;
    for(int i = 0; i < step; i++){
	//std::cout<<"step: "<<i<<std::endl;
        int row = i * batch_size, index;
	loss_sum = 0.0;
 	update_parameter();
	memset(temp_value, 0, data->glo_fea_dim);
	while( (row < (i + 1) * batch_size) && (row < data->fea_matrix.size()) ){
	    float wx = bias, p = 0.0, value;
	    for(int col = 0; col < data->fea_matrix[row].size(); col++){//for one instance
	  	index = data->fea_matrix[row][col].idx;
	        value = data->fea_matrix[row][col].val;
		temp_value[index] += value;
		//std::cout<<index<<" : "<<value<<std::endl;
	        wx += loc_w[index] * value;
	    }
	    //std::cout<<wx<<std::endl; 
            for(int k = 0; k < factor; k++){
		float vxvx = 0.0, vvxx = 0.0;
		for(int col = 0; col < data->fea_matrix[row].size(); col++){
		    index = data->fea_matrix[row][col].idx;
                    value = data->fea_matrix[row][col].val;
                    vxvx += loc_v_arr[col][k] * value;
		    vvxx += loc_v_arr[col][k] * loc_v_arr[col][k] * value*value;
                }
	        vxvx *= vxvx;
		vxvx -= vvxx;
	        //std::cout<<vxvx<<std::endl;
		wx += vxvx * 1.0 / 2;	
            }
	    //std::cout<<wx<<std::endl;
	    p = sigmoid(wx);
	    loss_sum += (p - data->label[row]);
	    //std::cout<<"loss_sum = "<<loss_sum<<std::endl; 
	    ++row;
    	}
	update_g();
        update_other_parameter();     
	std::cout<<"end one step"<<std::endl;
    }//end for
}

void FTRL::run(){
    ftrl();
}
