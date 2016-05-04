#include "mpi.h"
#include <iostream>
#include <vector>
#include "owlqn.h"

extern "C"{
#include <cblas.h>
}

LR::LR(Load_Data* ld, int total_num_proc, int my_rank) 
       : data(ld), num_proc(total_num_proc), rank(my_rank) {
    init();
}

LR::~LR(){
    delete[] glo_w; 
    delete[] glo_new_w;

    delete[] loc_g;
    delete[] glo_g;
    delete[] loc_new_g;
    delete[] glo_new_g;

    delete[] glo_sub_g;

    delete[] glo_q;

    for(int i = 0; i < m; i++){
        delete[] glo_s_list[i];
        delete[] glo_y_list[i];
    }
    delete[] glo_s_list;
    delete[] glo_y_list;

    delete[] glo_alpha_list;
    delete[] glo_ro_list;
}

void LR::init(){
    c = 1.0;

    glo_w = new double[data->glo_fea_dim]();
    glo_new_w = new double[data->glo_fea_dim]();

    loc_g = new double[data->glo_fea_dim]();
    glo_g = new double[data->glo_fea_dim]();
    loc_new_g = new double[data->glo_fea_dim]();
    glo_new_g = new double[data->glo_fea_dim]();

    glo_sub_g = new double[data->glo_fea_dim]();

    glo_q = new double[data->glo_fea_dim]();

    m = 10;
    now_m = 0;
    glo_s_list = new double*[m];
    for(int i = 0; i < m; i++){
        glo_s_list[i] = new double[data->glo_fea_dim]();
    }
    glo_y_list = new double*[m];
    for(int i = 0; i < m; i++){
        glo_y_list[i] = new double[data->glo_fea_dim]();
    }
    glo_alpha_list = new double[data->glo_fea_dim]();
    glo_ro_list = new double[data->glo_fea_dim](); 

    loc_loss = 0.0;
    glo_loss = 0.0;
    loc_new_loss = 0.0;
    glo_new_loss = 0.0;

    lambda = 1.0;
    backoff = 0.5;

    step = 0;
}

double LR::sigmoid(double x){
    return 1/(1+exp(-x));
}

double LR::calculate_loss(double *para_w){
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

void LR::calculate_gradient(){
    double f = 0.0;
    for(int i = 0; i < data->fea_matrix.size(); i++){
        int index;
        double wx = 0.0, value = 0.0;
        for(int j = 0; j <data->fea_matrix[i].size(); j++){
            index = data->fea_matrix[i][j].idx;
            value = data->fea_matrix[i][j].val;
            //std::cout<<"index="<<index<<std::endl;
            //std::cout<<"value="<<value<<std::endl;
            wx += *(glo_w + index) * value;
        }
        for(int j = 0; j < data->fea_matrix[i].size(); j++){
            //std::cout<<data->label[i]<<std::endl;
            *(glo_g + j) += (sigmoid(wx) - data->label[i]) * value / (1.0 * data->fea_matrix.size());
        }
    }
    //for(int i = 0; i < data->fea_matrix[i].size(); i++){
    //    std::cout<<*(para_g + i) <<std::endl;
    //}
}

void LR::calculate_subgradient(){
    if(c == 0.0){
        for(int j = 0; j < data->glo_fea_dim; j++){
            *(glo_sub_g + j) = -1 * *(glo_g + j);
        }
    }
    else if(c != 0.0){
        for(int j = 0; j < data->glo_fea_dim; j++){
            if(*(glo_w + j) > 0){
                *(glo_sub_g + j) = *(glo_g + j) + c;
            }
            else if(*(glo_w + j) < 0){
                *(glo_sub_g + j) = *(glo_g + j) - c;
            }
            else {
                if(*(glo_g + j) - c > 0) *(glo_sub_g + j) = *(glo_g + j) - c;//左导数
                else if(*(glo_g + j) + c < 0) *(glo_sub_g + j) = *(glo_g + j) + c;
                else *(glo_sub_g + j) = 0;
            }
            //std::cout<<*(local_sub_g + j)<<std::endl;
            //std::cout<<c<<std::endl;
        }
    }
}

void LR::fix_dir(){
    for(int j = 0; j < data->glo_fea_dim; j++){
        if(*(glo_new_w + j) * *(glo_w + j) >=0) *(glo_new_w + j) = 0.0;
        else *(glo_new_w + j) = *(glo_new_w + j);
    }
}

void LR::line_search(){
    while(true){
        for(int j = 0; j < data->glo_fea_dim; j++){
            *(glo_new_w + j) = *(glo_w + j) + lambda * *(glo_g + j);//local_g equal all nodes g
        }
        glo_new_loss = calculate_loss(glo_new_w);//cal new loss per thread
        if(glo_new_loss <= glo_loss + lambda * cblas_ddot(data->glo_fea_dim, (double*)glo_sub_g, 1, (double*)glo_g, 1)){
            break;
        }
        lambda *= backoff;
    }
}

void LR::two_loop(){
    cblas_dcopy(data->glo_fea_dim, glo_sub_g, 1, glo_q, 1);
    if(now_m > m) now_m = m; 
    for(int loop = now_m-1; loop >= 0; --loop){
        glo_ro_list[loop] = cblas_ddot(data->glo_fea_dim, &(*glo_y_list)[loop], 1, &(*glo_s_list)[loop], 1);
        glo_alpha_list[loop] = cblas_ddot(data->glo_fea_dim, &(*glo_s_list)[loop], 1, (double*)glo_q, 1) / glo_ro_list[loop];
        cblas_daxpy(data->glo_fea_dim, -1 * glo_alpha_list[loop], &(*glo_y_list)[loop], 1, (double*)glo_q, 1);
    }

    double ydoty = cblas_ddot(data->glo_fea_dim, glo_s_list[step%now_m - 1], 1, glo_y_list[step%now_m - 1], 1);
    float gamma = glo_ro_list[step%now_m - 1]/ydoty;
    cblas_dscal(data->glo_fea_dim, gamma, (double*)glo_q, 1);
    
    for(int loop = 0; loop < now_m; ++loop){
        double beta = cblas_ddot(data->glo_fea_dim, &(*glo_y_list)[loop], 1, (double*)glo_q, 1)/glo_ro_list[loop];
        cblas_daxpy(data->glo_fea_dim, glo_alpha_list[loop] - beta, &(*glo_s_list)[loop], 1, (double*)glo_q, 1);
    }
}

void LR::owlqn(){
    MPI_Status status;
    while(step < 3){
	//define and initial local parameters
	calculate_gradient();//calculate gradient of loss by global w)
	calculate_subgradient();
	two_loop();
	if(rank != 0){
	    MPI_Send(glo_q, data->glo_fea_dim, MPI_DOUBLE, 0, 2012, MPI_COMM_WORLD);
	}
	else if(rank == 0){
	    for(int j = 0; j < data->glo_fea_dim; j++){
	        *(glo_g + j) += *(glo_q + j);
	    }
            for(int i = 1; i < num_proc; i++){
                MPI_Recv(glo_q, data->glo_fea_dim, MPI_DOUBLE, MPI_ANY_SOURCE, 2012, MPI_COMM_WORLD, &status);
	        for(int j = 0; j < data->glo_fea_dim; j++){
                    *(glo_g + j) += *(glo_q + j);
	        }
            }
            glo_loss = calculate_loss(glo_w);
	    line_search();
            fix_dir();//orthant limited
	}
	//update slist
	cblas_daxpy(data->glo_fea_dim, -1, (double*)glo_w, 1, (double*)glo_new_w, 1);
	cblas_dcopy(data->glo_fea_dim, (double*)glo_new_w, 1, (double*)glo_s_list[(m - now_m) % m], 1);
	//update ylist
	cblas_daxpy(data->glo_fea_dim, -1, (double*)glo_g, 1, (double*)glo_new_g, 1);
	cblas_dcopy(data->glo_fea_dim, (double*)glo_new_g, 1, (double*)glo_y_list[(m - now_m) % m], 1);
	now_m++;
        if(now_m > m){
            for(int j = 0; j < data->glo_fea_dim; j++){
                 *(*(glo_s_list + abs(m - now_m) % m) + j) = 0.0;
                 *(*(glo_y_list + abs(m - now_m) % m) + j) = 0.0;
            }
        }
        cblas_dcopy(data->glo_fea_dim, (double*)glo_new_w, 1, (double*)glo_w, 1);
            step++;
        }
}

void LR::run(){
    owlqn();
}
