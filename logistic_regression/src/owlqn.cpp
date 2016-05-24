#include "mpi.h"
#include <iostream>
#include <vector>
#include <algorithm>
#include "owlqn.h"
#include <glog/logging.h>
#include <cstdlib>
#include <ctime>

#define NUM 999

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

    delete[] loc_z;

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
    srand(time(NULL));
    for(int i = 0; i < data->glo_fea_dim; i++) {
        glo_w[i] = rand() % (NUM+1) / (float)(NUM+1);
    }

    loc_z = new double[data->loc_ins_num]();
    calculate_z();

    loc_g = new double[data->glo_fea_dim]();
    glo_g = new double[data->glo_fea_dim]();
    loc_new_g = new double[data->glo_fea_dim]();
    glo_new_g = new double[data->glo_fea_dim]();

    glo_sub_g = new double[data->glo_fea_dim]();

    glo_q = new double[data->glo_fea_dim]();

    m = 10;
    now_m = 1;
    glo_s_list = new double*[m];
    for(int i = 0; i < m; i++){
        glo_s_list[i] = new double[data->glo_fea_dim]();
	for(int j = 0; j < data->glo_fea_dim; j++){
	    glo_s_list[i][j] = glo_w[j];
	}
    }
    glo_y_list = new double*[m];
    for(int i = 0; i < m; i++){
        glo_y_list[i] = new double[data->glo_fea_dim]();
	for(int j = 0; j < data->glo_fea_dim; j++){
            glo_y_list[i][j] = glo_g[j];
        }
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

void LR::calculate_z() {
    size_t idx = 0;
    double val = 0;
    for(int i = 0; i < data->loc_ins_num; i++) {
        loc_z[i] = 0;
        for(int j = 0; j < data->fea_matrix[i].size(); j++) {
            idx = data->fea_matrix[i][j].idx;
            val = data->fea_matrix[i][j].val;
            loc_z[i] += glo_w[idx] * val;
        }
    }
}

double LR::sigmoid(double x){
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

double LR::calculate_loss(double *para_w){
    double f = 0.0, val = 0.0, wx = 0.0, single_loss = 0.0;
    int index;
    for(int i = 0; i < data->fea_matrix.size(); i++){
        wx = 0.0;
        for(int j = 0; j < data->fea_matrix[i].size(); j++){
            index = data->fea_matrix[i][j].idx;
            val = data->fea_matrix[i][j].val;
            //LOG(INFO) << *(para_w + index) << std::endl;
            wx += *(para_w + index) * val;//maybe add bias later
        }
        //LOG(INFO)<<"wx: "<<sigmoid(wx)<<std::endl;
        single_loss = data->label[i] * log(sigmoid(wx)) + (1 - data->label[i]) * log(1 - sigmoid(wx));
        f += single_loss;
    }
    return -f / data->fea_matrix.size();
}

void LR::calculate_gradient(){
    int index, single_feature_num;
    double value;
    int instance_num = data->fea_matrix.size();
    LOG(INFO) << "process " << rank << ", instance num " << instance_num << std::endl;
    for(int i = 0; i < instance_num; i++){
        single_feature_num = data->fea_matrix[i].size();
        for(int j = 0; j < single_feature_num; j++){
            index = data->fea_matrix[i][j].idx;
            value = data->fea_matrix[i][j].val;
            loc_g[index] += (sigmoid(loc_z[i]) - data->label[i]) * value;
            DLOG(INFO) << "loc_g[" << index << "]: " << loc_g[index] << " after instance " << i + 1  << "/" << instance_num << " in rank " << rank <<std::endl << std::flush;
        }
	/*
	if(i == instance_num - 1){
	    for(int index = 0; index < data->glo_fea_dim; index++)
	    std::cout<< "loc_g[" << index << "]: " << loc_g[index] << " after instance " << i + 1  << "/" << instance_num << " in rank " << rank <<std::endl << std::flush;
	
	}*/
    }
    for(int index = 0; index < data->glo_fea_dim; index++){
	glo_g[index] = loc_g[index] / instance_num;
	//std::cout<< "glo_g[" << index << "]: " << glo_g[index] << " after normal " << "in rank " << rank <<std::endl << std::flush;	
    }
}

void LR::calculate_subgradient(){
    if(c == 0.0){
        for(int j = 0; j < data->glo_fea_dim; j++){
            *(glo_sub_g + j) = -1 * *(glo_g + j);
        }
    }
    else if(c != 0.0){
        for(int j = 0; j < data->glo_fea_dim; j++){
            //LOG(INFO) << *(glo_g + j) << std::endl;
            //LOG(INFO) << *(glo_w + j) << std::endl;
            if(*(glo_w + j) > 0){
                *(glo_sub_g + j) = *(glo_g + j) + c;
            }
            else if(*(glo_w + j) < 0){
                *(glo_sub_g + j) = *(glo_g + j) - c;
            }
            else {
                //LOG(INFO) << *(glo_g + j) - c << std::endl;
                if(*(glo_g + j) - c > 0) *(glo_sub_g + j) = *(glo_g + j) - c;//左导数
                else if(*(glo_g + j) + c < 0) *(glo_sub_g + j) = *(glo_g + j) + c;
                else *(glo_sub_g + j) = 0;
            }
            //LOG(INFO) << *(glo_sub_g + j) << std::endl;
            //LOG(INFO) << c <<std::endl;
        }
    }
    /*
    for(int i = 0; i < data->glo_fea_dim; i++){
	if(rank == 0)
	std::cout<<"glo_sub_g["<<i<<"]: "<<glo_sub_g[i]<<" in rank:" <<rank<<std::endl;
    }*/
}

void LR::fix_dir_glo_q(){
    /*
    for(int j = 0; j < data->glo_fea_dim; j++){
	//if(rank == 0) std::cout<<"glo_q["<<j<<"]"<<glo_q[j]<<std::endl;
	if(rank == 0) std::cout<<"glo_sub_g["<<j<<"]"<<glo_sub_g[j]<<std::endl;
    }
    */
    for(int j = 0; j < data->glo_fea_dim; ++j){
	if(*(glo_q + j) * *(glo_sub_g +j) >= 0){
   	    *(glo_q + j) = 0.0;
        }
	else{
	    glo_q[j] = glo_q[j]; 
	}
    }
    /*
    for(int j = 0; j < data->glo_fea_dim; ++j){
	std::cout<<"glo_q["<<j<<"]: "<<glo_q[j]<<std::endl;
    }
    */
}

void LR::fix_dir_glo_new_w(){
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
        loc_new_loss = calculate_loss(glo_new_w);//cal new loss per thread
        //LOG(INFO) << "masterid:" << MASTER_ID << std::endl;
        //LOG(INFO) << "before reduce rank:" << rank <<" loc_new_loss:" << loc_new_loss << " glo_loss:" << glo_loss << std::endl;
        MPI_Allreduce(&loc_new_loss, &glo_new_loss, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        //LOG(INFO) << "after reduce rank:" << rank << " glo_new_loss:" << glo_new_loss << " glo_loss:" << glo_loss << std::endl;
        if(glo_new_loss <= glo_loss + lambda * cblas_ddot(data->glo_fea_dim, (double*)glo_sub_g, 1, (double*)glo_g, 1)){
            break;
        }
        lambda *= backoff;
        //LOG(INFO) << lambda << std::endl;
        if(lambda <= 1e-6) break;
    }
}

void LR::two_loop(){
    cblas_dcopy(data->glo_fea_dim, glo_sub_g, 1, glo_q, 1);
    /*for(int i = 0; i < data->glo_fea_dim; i++){
	if(rank == 0) std::cout<<glo_q[i]<<std::endl;
    }*/
    if(now_m > m) now_m = m; 
    std::cout<<"now_m "<<now_m<<std::endl;
    for(int loop = now_m-1; loop >= 0; --loop){
        glo_ro_list[loop] = cblas_ddot(data->glo_fea_dim, &(*glo_y_list)[loop], 1, &(*glo_s_list)[loop], 1);
	/*
	for(int i = 0; i < now_m; i++){
            for(int j = 0; j < data->glo_fea_dim; j++){
	        if(rank == 0) std::cout<<"glo_s_list["<<i<<"] = "<<glo_s_list[i][j]<<" ";
   	    }
	    std::cout<<std::endl;
        }*/
        glo_alpha_list[loop] = cblas_ddot(data->glo_fea_dim, &(*glo_s_list)[loop], 1, (double*)glo_q, 1) / (glo_ro_list[loop] + 1.0);
        //std::cout<<glo_alpha_list[loop]<<std::endl;
        cblas_daxpy(data->glo_fea_dim, -1 * glo_alpha_list[loop], &(*glo_y_list)[loop], 1, (double*)glo_q, 1);
	/*for(int i = 0; i < data->glo_fea_dim; i++){
            if(rank == 0) std::cout<<glo_q[i]<<std::endl;
        }*/
    }
    //std::cout<<step<<std::endl;
    if(step != 0){
        double ydoty = cblas_ddot(data->glo_fea_dim, glo_s_list[now_m - 1], 1, glo_y_list[now_m - 1], 1);
        float gamma = glo_ro_list[now_m - 1]/ydoty;
        cblas_dscal(data->glo_fea_dim, gamma, (double*)glo_q, 1);
    }
    //std::cout<<step<<std::endl;
    for(int loop = 0; loop < now_m; ++loop){
        double beta = cblas_ddot(data->glo_fea_dim, &(*glo_y_list)[loop], 1, (double*)glo_q, 1)/(glo_ro_list[loop] + 1.0);
	//std::cout<<"beta "<<beta<<std::endl;
        cblas_daxpy(data->glo_fea_dim, glo_alpha_list[loop] - beta, &(*glo_s_list)[loop], 1, (double*)glo_q, 1);
    }
    /*
    for(int i = 0; i < data->glo_fea_dim; i++){
        if(rank == 0) std::cout<<glo_q[i]<<std::endl;
    }
    */
}

void LR::update_state(){

     //update lbfgs memory
    update_memory();//not distributed

    //update w
    std::swap(glo_w, glo_new_w); 

    //update loss
    glo_loss = glo_new_loss;

    //update z
    calculate_z();

    //update step count
    step++;
}
void LR::update_memory(){
    //update slist
    cblas_daxpy(data->glo_fea_dim, -1, (double*)glo_w, 1, (double*)glo_new_w, 1);
    cblas_dcopy(data->glo_fea_dim, (double*)glo_new_w, 1, (double*)glo_s_list[now_m % m], 1);
    //update ylist
    cblas_daxpy(data->glo_fea_dim, -1, (double*)glo_g, 1, (double*)glo_new_g, 1);
    cblas_dcopy(data->glo_fea_dim, (double*)glo_new_g, 1, (double*)glo_y_list[now_m % m], 1);
    now_m++;
}

bool LR::meet_criterion(){
    if(step == 30) return true;
    return false;
}

void LR::owlqn(){
    while(true){
        calculate_gradient(); //distributed, calculate gradient is distributed
        LOG(INFO) << "process " << rank << " calculate gradient over" << std::endl << std::flush;
        calculate_subgradient(); //not distributed, only on master process
 	//std::cout<<"-------------------------------------------"<<std::endl;
        LOG(INFO) << "process " << rank << " calculate sub-gradient over" << std::endl << std::flush;
        two_loop();//not distributed, only on master process
        LOG(INFO) << "process " << rank << " calculate two-loop over" << std::endl << std::flush;
        fix_dir_glo_q();//not distributed, orthant limited
        LOG(INFO) << "process " << rank << " fix-dir over" << std::endl << std::flush;
        line_search();//distributed, calculate loss is distributed
	fix_dir_glo_new_w();
        std::cout<<step<<std::endl;
        if(meet_criterion()) {//not distributed
	    //std::cout<<step<<std::endl;
            break;
        } else {
            LOG(INFO) << "process " << rank << " step " << step << std::endl << std::flush;
            update_state();
        }
    }
    for(int j = 0; j < data->glo_fea_dim; j++){
	std::cout<<"glow["<<j<<"] "<<glo_w[j]<<std::endl;
    }
}

void LR::run(){
    owlqn();
}
