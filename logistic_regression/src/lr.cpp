#include "mpi.h"
#include <iostream>
#include <vector>
#include "lr.h"

extern "C"{
#include <cblas.h>
}

LR::LR(Load_Data* ld) : data(ld){
    init();
}

LR::~LR(){
    delete [] ro_list;
    for(int i = 0; i < m; i++){
        free(s_list[i]);
        free(y_list[i]);
    }
    free(s_list);
    free(y_list);
    delete [] w; 
    delete [] next_w;
    delete [] g;
}

void LR::init(){
    c = 1.0;
    m = 10;
    lambda = 1.0;

    w = new double[data->fea_dim];
    next_w = new double[data->fea_dim];
    global_w = new double[data->fea_dim];
    g = new double[data->fea_dim];

    old_loss = 0.0;
    new_loss = 0.0;
    q = new double[data->fea_dim];//local variable
    alpha = (double*)malloc(sizeof(double)*data->fea_dim);

    ro_list = new double[data->fea_dim];
    s_list = (double**)malloc(sizeof(double*)*m);
    for(int i = 0; i < m; i++){
        s_list[i] = (double*)malloc(sizeof(double)*data->fea_dim);
    }

    y_list = (double**)malloc(sizeof(double*)*m);
    for(int i = 0; i < m; i++){
        y_list[i] = (double*)malloc(sizeof(double)*data->fea_dim);
    }
 
    double init_w = 0.0;
    for(int j = 0; j < data->fea_dim; j++){
        *(w + j) = init_w;
        *(next_w + j) = init_w;
        *(global_w + j) = init_w; 
        *(g + j) = init_w;
    }
}

double LR::sigmoid(double x){
    return 1/(1+exp(-x));
}

double LR::loss_function_value(double *para_w){
    double f = 0.0;
    for(int i = 0; i < data->fea_matrix.size(); i++){
        double wx = 0.0;
        for(int j = 0; j < data->fea_matrix[i].size(); j++){
            int id = data->fea_matrix[i][j].idx;
            double val = data->fea_matrix[i][j].val;
            wx += -*(para_w + id) * val;//maybe add bias later
        }
        double l = data->label[i] * log(sigmoid(wx)) + (1 - data->label[i]) * log(1 - sigmoid(wx));
        f += l;
    }
    return f / data->fea_matrix.size();
}

void LR::calculate_gradient(double *w, double *g){
    double f = 0.0;
    //std::cout<<data->fea_matrix.size()<<"----"<<std::endl;
    for(int i = 0; i < data->fea_matrix.size(); i++){
        int index;
        double wx = 0.0, value = 0.0;
        for(int j = 0; j <data->fea_matrix[i].size(); j++){
            index = data->fea_matrix[i][j].idx;
            value = data->fea_matrix[i][j].val;
            //std::cout<<"index="<<index<<std::endl;
            //std::cout<<"value="<<value<<std::endl;
            wx += *(w + index) * value;
        }
        for(int j = 0; j < data->fea_matrix[i].size(); j++){
            //std::cout<<data->label[i]<<std::endl;
            *(g + j) += (sigmoid(wx) - data->label[i]) * value / (1.0 * data->fea_matrix.size());
        }
    }
    //for(int i = 0; i < data->fea_matrix[i].size(); i++){
    //    std::cout<<*(para_g + i) <<std::endl;
    //}
}

void LR::calculate_subgradient(double * g, double *sub_g){
    if(c == 0.0){
        for(int j = 0; j < data->fea_dim; j++){
            *(sub_g + j) = -1 * *(g + j);
        }
    }
    else if(c != 0.0){
        for(int j = 0; j < data->fea_dim; j++){
            if(*(w + j) > 0){
                *(sub_g + j) = *(g + j) + c;
            }
            else if(*(w + j) < 0){
                *(sub_g + j) = *(g + j) - c;
            }
            else {
                if(*(g + j) - c > 0) *(sub_g + j) = *(g + j) - c;//左导数
                else if(*(g + j) + c < 0) *(sub_g + j) = *(g + j) + c;
                else *(sub_g + j) = 0;
            }
            //std::cout<<*(local_sub_g + j)<<std::endl;
            //std::cout<<c<<std::endl;
        }
    }
}

void LR::fix_dir(double *w, double *next_w){
    for(int j = 0; j < data->fea_dim; j++){
        if(*(next_w + j) * *(w + j) >=0) *(next_w + j) = 0.0;
        else *(next_w + j) = *(next_w + j);
    }
}

void LR::line_search(){
    double backoff = 0.5;
    while(true){
        if(rank != 0){
 	    //send old_loss
        }
	else{
           // 
	}
        for(int j = 0; j < data->fea_dim; j++){
            *(next_w + j) = *(w + j) + lambda * *(global_g + j);//local_g equal all nodes g
        }
        new_loss = calculate_loss(next_w);//cal new loss per thread
        if(rank != 0){
            //send old_loss
        }
        else{
           //
        }

        if(new_loss <= loss + lambda * cblas_ddot(data->fea_dim, (double*)sub_g, 1, (double*)global_g, 1)){
            break;
        }
        lambda *= backoff;
    }
}

void LR::two_loop(){
    cblas_dcopy(data->fea_dim, sub_g, 1, q, 1);
    if(use_list_len < m) m = use_list_len; 
    for(int loop = m-1; loop >= 0; --loop){
        ro_list[loop] = cblas_ddot(data->fea_dim, &(*y_list)[loop], 1, &(*s_list)[loop], 1);
        alpha[loop] = cblas_ddot(data->fea_dim, &(*s_list)[loop], 1, (double*)q, 1) / ro_list[loop];
        cblas_daxpy(data->fea_dim, -1 * alpha[loop], &(*y_list)[loop], 1, (double*)q, 1);
    }

    double ydoty = cblas_ddot(data->fea_dim, s_list[step%m - 1], 1, y_list[step%m - 1], 1);
    float gamma = ro_list[step%m - 1]/ydoty;
    cblas_sscal(data->fea_dim, gamma, p, 1);
    
    for(int loop = 0; loop < m; ++loop){
        double beta = cblas_ddot(data->fea_dim, &(*y_list)[loop], 1, (double*)p, 1)/ro_list[loop];
        cblas_daxpy(data->fea_dim, alpha[loop] - beta, &(*s_list)[loop], 1, (double*)p, 1);
    }
    delete [] alpha;
}

void LR::owlqn(int rank, int n_proc){
    MPI_Status status;
    int use_list_len = 0;
    int step = 0;
    while(step < 3){
	//define and initial local parameters
	calculate_gradient();//calculate gradient of loss by global w)
	calculate_subgradient();
	two_loop();
	if(rank != 0){
	    MPI_Send(q, data->fea_dim, MPI_DOUBLE, 0, 2012, MPI_COMM_WORLD);
	}
	else if(rank == 0){
	    for(int j = 0; j < data->fea_dim; j++){
	        *(global_g + j) += *(q + j);
	    }
            for(int i = 1; i < nproc; i++){
                MPI_Recv(q, data->fea_dim, MPI_DOUBLE, MPI_ANY_SOURCE, 2012, MPI_COMM_WORLD, &status);
	        for(int j = 0; j < data->fea_dim; j++){
                    *(global_g + j) += *(q + j);
	        }
            }
            loss = calculate_loss(w);
	    line_search();
            fix_dir(w, next_w);//orthant limited
	}
	//update slist
	cblas_daxpy(data->fea_dim, -1, (double*)w, 1, (double*)next_w, 1);
	cblas_dcopy(data->fea_dim, (double*)next_w, 1, (double*)s_list[(m - use_list_len) % m], 1);
	//update ylist
	cblas_daxpy(data->fea_dim, -1, (double*)global_g, 1, (double*)global_next_g, 1);
	cblas_dcopy(data->fea_dim, (double*)global_next_g, 1, (double*)y_list[(m - use_list_len) % m], 1);
	use_list_len++;
        if(use_list_len > m){
            for(int j = 0; j < data->fea_dim; j++){
                 *(*(s_list + abs(m - use_list_len) % m) + j) = 0.0;
                 *(*(y_list + abs(m - use_list_len) % m) + j) = 0.0;
            }
        }
        cblas_dcopy(data->fea_dim, (double*)next_w, 1, (double*)w, 1);
            step++;
        }
}

void LR::run(int rank, int nproc){
    owlqn(rank, nproc);
}
