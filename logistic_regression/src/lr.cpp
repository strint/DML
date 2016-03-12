#include "mpi.h"
#include <iostream>
#include <vector>
#include "lr.h"

extern "C"{
#include <cblas.h>
}

LR::LR(){
}

LR::~LR(){
    delete [] w; 
    delete [] next_w;
    delete [] global_g;
    delete [] global_next_g;
    delete [] all_nodes_global_g;
}

void LR::init_theta(){
    c = 1.0;
    m = 10;
    w = new double[feature_dim];
    next_w = new double[feature_dim];
    global_g = new double[feature_dim];
    global_next_g = new double[feature_dim];
    all_nodes_global_g = new double[feature_dim];

    global_old_loss_val = 0.0;
    global_new_loss_val = 0.0;
    
    main_thread_id = pthread_self();   
    pthread_barrier_init(&barrier, NULL, 2);
 
    data = NULL;
    double init_w = 0.0;
    for(int j = 0; j < feature_dim; j++){
        *(w + j) = init_w;
        *(next_w + j) = init_w;
        *(global_g + j) = init_w;
        *(global_next_g + j) = init_w; 
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

void LR::loss_function_gradient(double *para_w, double *para_g){
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
            wx += *(para_w + index) * value;
        }
        for(int j = 0; j < data->fea_matrix[i].size(); j++){
            //std::cout<<data->label[i]<<std::endl;
            *(para_g + j) += (sigmoid(wx) - data->label[i]) * value / (1.0 * data->fea_matrix.size());
        }
    }
    //for(int i = 0; i < data->fea_matrix[i].size(); i++){
    //    std::cout<<*(para_g + i) <<std::endl;
    //}
}

void LR::loss_function_subgradient(double * local_g, double *local_sub_g){
    if(c == 0.0){
        for(int j = 0; j < feature_dim; j++){
            *(local_sub_g + j) = -1 * *(local_g + j);
        }
    }
    else if(c != 0.0){
        for(int j = 0; j < feature_dim; j++){
            if(*(w + j) > 0){
                *(local_sub_g + j) = *(local_g + j) + c;
            }
            else if(*(w + j) < 0){
                *(local_sub_g + j) = *(local_g + j) - c;
            }
            else {
                if(*(local_g + j) - c > 0) *(local_sub_g + j) = *(local_g + j) - c;//左导数
                else if(*(local_g + j) + c < 0) *(local_sub_g + j) = *(local_g + j) + c;
                else *(local_sub_g + j) = 0;
            }
            //std::cout<<*(local_sub_g + j)<<std::endl;
            //std::cout<<c<<std::endl;
        }
    }
}

void LR::fix_dir(double *w, double *next_w){
    for(int j = 0; j < feature_dim; j++){
        if(*(next_w + j) * *(w + j) >=0) *(next_w + j) = 0.0;
        else *(next_w + j) = *(next_w + j);
    }
}

void LR::line_search(double *param_g){
    double alpha = 1.0;
    double beta = 1e-4;
    double backoff = 0.5;
    double old_loss_val = 0.0, new_loss_val = 0.0;
    while(true){
        old_loss_val = loss_function_value(w);//cal loss value per thread
        //std::cout<<old_loss_val<<std::endl;    
        pthread_mutex_lock(&mutex);
        global_old_loss_val += old_loss_val;//add old loss value of all threads
        pthread_mutex_unlock(&mutex); 
        //std::cout<<global_old_loss_val<<std::endl;
        pid_t local_thread_id;
        local_thread_id = pthread_self();
        if(local_thread_id == main_thread_id){
            MPI_Allreduce(&global_old_loss_val, &all_nodes_old_loss_val, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
        }
        pthread_barrier_wait(&barrier);
        for(int j = 0; j < feature_dim; j++){
            *(next_w + j) = *(w + j) + alpha * *(param_g + j);//local_g equal all nodes g
        }
        fix_dir(w, next_w);//orthant limited
        new_loss_val = loss_function_value(next_w);//cal new loss per thread
        pthread_mutex_lock(&mutex);
        global_new_loss_val += new_loss_val;//sum all threads loss value
        pthread_mutex_unlock(&mutex);
        //if(lr.rank != 0){
        if(rank != 0){
            if(local_thread_id == main_thread_id){
                MPI_Allreduce(&global_new_loss_val, &all_nodes_new_loss_val, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);//sum all nodes loss.
            }
        }
        pthread_barrier_wait(&barrier);
        loss_function_gradient(next_w, global_next_g);

        if(all_nodes_new_loss_val <= all_nodes_old_loss_val + beta * cblas_ddot(feature_dim, (double*)param_g, 1, (double*)global_next_g, 1)){
            break;
        }
        alpha *= backoff;
    }
}

void LR::two_loop(int step, int use_list_len, double *local_sub_g, double **s_list, double **y_list, double *ro_list, float *p){
    double *q = new double[feature_dim];//local variable
    double* alpha = (double*)malloc(sizeof(double)*feature_dim);
    /*for(int i = 0; i < feature_dim; i++){
        //std::cout<<*(local_sub_g + i) << std::endl;
        std::cout<<*(p+i)<<std::endl;
    }*/
    //free(p);
    cblas_dcopy(feature_dim, local_sub_g, 1, q, 1);
    /*for(int i = 0; i < feature_dim; i++){
        std::cout<<*(q + i) << std::endl;
    }*/
    if(use_list_len < m) m = use_list_len; 
    for(int loop = 1; loop <= m; ++loop){
        ro_list[loop - 1] = cblas_ddot(feature_dim, &(*y_list)[loop - 1], 1, &(*s_list)[loop - 1], 1);
        alpha[loop] = cblas_ddot(feature_dim, &(*s_list)[loop - 1], 1, (double*)q, 1)/ro_list[loop - 1];
        cblas_daxpy(feature_dim, -1 * alpha[loop], &(*y_list)[loop - 1], 1, (double*)q, 1);
    }
    if(step == 0){//the first step, p should be unit vector;
        for(int j = 0; j < feature_dim; j++){
            *(p + j) = 1.0;
        }
    }
    else if(step != 0){  
        double ydoty = cblas_ddot(feature_dim, s_list[step-1], 1, y_list[step-1], 1);
        float gamma = ro_list[step - 1]/ydoty;
        cblas_sscal(feature_dim, gamma, p, 1);
    }
    for(int loop = m; loop >=1; --loop){
        double beta = cblas_ddot(feature_dim, &(*y_list)[m - loop], 1, (double*)p, 1)/ro_list[m - loop];
        cblas_daxpy(feature_dim, alpha[loop] - beta, &(*s_list)[m - loop], 1, (double*)p, 1);
    }
    delete [] alpha;
    //std::cout<<11111<<std::endl;
}

void LR::parallel_owlqn(int step, int use_list_len, double* ro_list, double** s_list, double** y_list){
    //define and initial local parameters
    double *local_g = (double*)malloc(sizeof(double)*feature_dim);//single thread gradient
    double *local_sub_g = new double[feature_dim];//single thread subgradient
    float *p = new float[feature_dim];//single thread search direction.after two loop
    loss_function_gradient(w, local_g);//calculate gradient of loss by global w)
    loss_function_subgradient(local_g, local_sub_g); 
    //should add code update multithread and all nodes sub_g to global_sub_g
    two_loop(step, use_list_len, local_sub_g, s_list, y_list, ro_list, p);
    pthread_mutex_lock(&mutex);
    for(int j = 0; j < feature_dim; j++){
        *(global_g + j) += *(p + j);//update global direction of all threads
    }
    pthread_mutex_unlock(&mutex);
    pid_t local_thread_id;
    local_thread_id = pthread_self();
    //if(lr.rank == 0){
    if(rank != 0){
        if(local_thread_id == main_thread_id){
            MPI_Send(global_g, feature_dim, MPI_DOUBLE, 0, 2012, MPI_COMM_WORLD); 
        }
    }
    else if(rank == 0){
        if(local_thread_id == main_thread_id){
            MPI_Status status;
            double* tmp_global_g = new double[feature_dim];
            MPI_Recv(global_g, feature_dim, MPI_DOUBLE, MPI_ANY_SOURCE, 2012, MPI_COMM_WORLD, &status);
        }
    }
    pthread_barrier_wait(&barrier);
    //should be synchronous all threads
    line_search(all_nodes_global_g);//use global search direction to search
    //update slist
    if(local_thread_id == main_thread_id){
        cblas_daxpy(feature_dim, -1, (double*)w, 1, (double*)next_w, 1);
        cblas_dcopy(feature_dim, (double*)next_w, 1, (double*)s_list[(m - use_list_len) % m], 1);
    //update ylist
        cblas_daxpy(feature_dim, -1, (double*)global_g, 1, (double*)global_next_g, 1); 
        cblas_dcopy(feature_dim, (double*)global_next_g, 1, (double*)y_list[(m - use_list_len) % m], 1);

        use_list_len++;
        if(use_list_len > m){
            for(int j = 0; j < feature_dim; j++){
                *(*(s_list + abs(m - use_list_len) % m) + j) = 0.0;
                *(*(y_list + abs(m - use_list_len) % m) + j) = 0.0;        
            }
        }
        cblas_dcopy(feature_dim, (double*)next_w, 1, (double*)w, 1);
    }
    pthread_barrier_wait(&barrier);
}

void LR::owlqn(int proc_id, int n_procs){
    std::cout<<proc_id<<"---"<<n_procs<<std::endl;
    double *ro_list = new double[feature_dim];
    double** s_list = (double**)malloc(sizeof(double*)*m);
    s_list[0] = new double[feature_dim];
    for(int i = 0; i < m; i++){
        s_list[i] = (double*)malloc(sizeof(double)*feature_dim); 
    }
    double** y_list = (double**)malloc(sizeof(double*)*m);
    for(int i = 0; i < m; i++){
        y_list[i] = (double*)malloc(sizeof(double)*feature_dim); 
    }
    int use_list_len = 0;
    int step = 0;
    while(step < 3){
        parallel_owlqn(step, use_list_len, ro_list, s_list, y_list);        
        step++;
    }
    pthread_barrier_wait(&barrier);
    delete [] ro_list;
    for(int i = 0; i < m; i++){
        free(s_list[i]);
        free(y_list[i]);
    }
    free(s_list);
    free(y_list);
}

