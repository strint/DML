#include "mpi.h"
#include <iostream>
#include <vector>
#include "lr.h"
#include "config.h"

extern "C"{
#include <cblas.h>
}

pthread_mutex_t LR::mutex;
pthread_barrier_t LR::barrier;

LR::LR(){
}

LR::~LR(){
    delete [] w;
    delete [] next_w;
    delete [] global_g;
    delete [] global_next_g;
    delete [] all_nodes_global_g;
}

void LR::init_thread_var(int threads_num) {
    pthread_mutex_init(&mutex, NULL);
    pthread_barrier_init(&barrier, NULL, threads_num);
}

void LR::destroy_thread_var() {
    pthread_barrier_destroy(&barrier);
    pthread_mutex_destroy(&mutex);
}

void LR::init_theta(){
    c = 1.0;
    m = 10;
    w = new float[train_data->fea_dim];
    next_w = new float[train_data->fea_dim];
    global_g = new float[train_data->fea_dim];
    global_next_g = new float[train_data->fea_dim];
    all_nodes_global_g = new float[train_data->fea_dim];

    global_old_loss_val = 0.0;
    global_new_loss_val = 0.0;

    main_thread_id = pthread_self();

    float init_w = 0.0;
    for(int j = 0; j < train_data->fea_dim; j++){
        *(w + j) = init_w;
        *(next_w + j) = init_w;
        *(global_g + j) = init_w;
        *(global_next_g + j) = init_w;
    }
}

//----------------------------owlqn--------------------------------------------
float LR::sigmoid(float x){
    float sgm = 1/(1+exp(-(float)x));
    return (float)sgm;
}

float LR::loss_function_value(float *para_w){
    float f = 0.0;
    for(int i = 0; i < train_data->fea_matrix.size(); i++){
        float x = 0.0;
        for(int j = 0; j < train_data->fea_matrix[i].size(); j++){
            int id = train_data->fea_matrix[i][j].idx;
            float val = train_data->fea_matrix[i][j].val;
            x += *(para_w + id) * val;//maybe add bias later
        }
        float l = train_data->label[i] * log(1/sigmoid(-1 * x)) + (1 - train_data->label[i]) * log(1/sigmoid(x));
        f += l;
    }
    return f;
}

void LR::loss_function_gradient(float *para_w, float *para_g){
    float f = 0.0;
    for(int i = 0; i < train_data->fea_matrix.size(); i++){
        float x = 0.0, value = 0.0;
        int index;
        for(int j = 0; j <train_data->fea_matrix[i].size(); j++){
            index = train_data->fea_matrix[i][j].idx;
            value = train_data->fea_matrix[i][j].val;
            x += *(para_w + index) * value;
        }
        for(int j = 0; j < train_data->fea_matrix[i].size(); j++){
            *(para_g + j) += train_data->label[i] * sigmoid(x) * value + (1 - train_data->label[i]) * sigmoid(x) * value;
        }
    }
    for(int j = 0; j < train_data->fea_dim; j++){
        *(para_g + j) /= train_data->fea_matrix.size();
    }
}

void LR::loss_function_subgradient(float * local_g, float *local_sub_g){
    if(c == 0.0){
        for(int j = 0; j < train_data->fea_dim; j++){
            *(local_sub_g + j) = -1 * *(local_g + j);
        }
    }
    else{
        for(int j = 0; j < train_data->fea_dim; j++){
            if(*(w + j) > 0){
                *(local_sub_g + j) = *(local_g + j) - c;
            }
            else if(*(w + j) < 0){
                *(local_sub_g + j) = *(local_g + j) - c;
            }
            else {
                if(*(local_g + j) - c > 0) *(local_sub_g + j) = c - *(local_g + j);
                else if(*(local_g + j) - c < 0) *(local_sub_g + j) = *(local_g + j) - c;
                else *(local_sub_g + j) = 0;
            }
        }
    }
}

void LR::fix_dir(float *w, float *next_w){
    for(int j = 0; j < train_data->fea_dim; j++){
        if(*(next_w + j) * *(w + j) >=0) *(next_w + j) = 0.0;
        else *(next_w + j) = *(next_w + j);
    }
}

void LR::line_search(float *param_g){
    float alpha = 1.0;
    float beta = 1e-4;
    float backoff = 0.5;
    float old_loss_val = 0.0, new_loss_val = 0.0;
    while(true){
        old_loss_val = loss_function_value(w);//cal loss value per thread

        pthread_mutex_lock(&mutex);
        global_old_loss_val += old_loss_val;//add old loss value of all threads
        pthread_mutex_unlock(&mutex);

        pid_t local_thread_id;
        local_thread_id = pthread_self();
        if(local_thread_id == main_thread_id){
            MPI_Allreduce(&global_old_loss_val, &all_nodes_old_loss_val, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
        }

        pthread_barrier_wait(&barrier);
        for(int j = 0; j < train_data->fea_dim; j++){
            *(next_w + j) = *(w + j) + alpha * *(param_g + j);//local_g equal all nodes g
        }
        fix_dir(w, next_w);//orthant limited
        new_loss_val = loss_function_value(next_w);//cal new loss per thread

        pthread_mutex_lock(&mutex);
        global_new_loss_val += new_loss_val;//sum all threads loss value
        pthread_mutex_unlock(&mutex);

        if(local_thread_id == main_thread_id){
            MPI_Allreduce(&global_new_loss_val, &all_nodes_new_loss_val, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);//sum all nodes loss.
        }

        pthread_barrier_wait(&barrier);
        loss_function_gradient(next_w, global_next_g);

        if(all_nodes_new_loss_val <= all_nodes_old_loss_val + beta * cblas_ddot(train_data->fea_dim, (double*)param_g, 1, (double*)global_next_g, 1)){
            break;
        }
        alpha *= backoff;
        break;
    }
}

void LR::two_loop(int use_list_len, float *local_sub_g, float **s_list, float **y_list, float *ro_list, float *p){
    std::cout << thread_rank << " owlqn h 1" << std::endl;
    float *alpha = new float[m];
    float *q = new float[train_data->fea_dim];//local variable
    cblas_dcopy(train_data->fea_dim, (double*)local_sub_g, 1, (double*)q, 1);
    if(use_list_len < m) m = use_list_len;

    for(int loop = 1; loop <= m; ++loop){
        ro_list[loop - 1] = cblas_ddot(train_data->fea_dim, (double*)(&(*y_list)[loop - 1]), 1, (double*)(&(*s_list)[loop - 1]), 1);
        alpha[loop] = cblas_ddot(train_data->fea_dim, (double*)(&(*s_list)[loop - 1]), 1, (double*)q, 1)/ro_list[loop - 1];
        cblas_daxpy(train_data->fea_dim, -1 * alpha[loop], (double*)(&(*y_list)[loop - 1]), 1, (double*)q, 1);
    }
    std::cout << thread_rank << " owlqn h 2" << std::endl;
    delete [] q;
    std::cout << thread_rank << " owlqn h 3" << std::endl;
    float *last_y = new float[train_data->fea_dim];
    for(int j = 0; j < train_data->fea_dim; j++){
        last_y[j] = *((*y_list + m - 1) + j);
    }

    float ydoty = cblas_ddot(train_data->fea_dim, (double*)last_y, 1, (double*)last_y, 1);
    float gamma = ro_list[m - 1]/ydoty;
    cblas_sscal(train_data->fea_dim, gamma,(float*)p, 1);

    for(int loop = m; loop >=1; --loop){
        float beta = cblas_ddot(train_data->fea_dim, (double*)(&(*y_list)[m - loop]), 1, (double*)p, 1)/ro_list[m - loop];
        cblas_daxpy(train_data->fea_dim, alpha[loop] - beta, (double*)(&(*s_list)[m - loop]), 1, (double*)p, 1);
    }
    delete [] alpha;
    delete [] last_y;
}

void LR::parallel_owlqn(int use_list_len, float* ro_list, float** s_list, float** y_list){
    std::cout << thread_rank << " owlqn e" << std::endl;
    //define and initial local parameters
    float *local_g = new float[train_data->fea_dim];//single thread gradient
    float *local_sub_g = new float[train_data->fea_dim];//single thread subgradient
    float *p = new float[train_data->fea_dim];//single thread search direction.after two loop
    std::cout << thread_rank << " owlqn f" << std::endl;
    loss_function_gradient(w, local_g);//calculate gradient of loss by global w)
    std::cout << thread_rank << " owlqn g" << std::endl;
    loss_function_subgradient(local_g, local_sub_g);
    std::cout << thread_rank << " owlqn h" << std::endl;
    //should add code update multithread and all nodes sub_g to global_sub_g
    two_loop(use_list_len, local_sub_g, s_list, y_list, ro_list, p);

    std::cout << thread_rank << " owlqn i" << std::endl;
    pthread_mutex_lock(&mutex);
    std::cout << thread_rank << " owlqn j" << std::endl;
    for(int j = 0; j < train_data->fea_dim; j++){
        *(global_g + j) += *(p + j);//update global direction of all threads
    }
    std::cout << thread_rank << " owlqn k" << std::endl;
    pthread_mutex_unlock(&mutex);

    pid_t local_thread_id;
    local_thread_id = pthread_self();
    std::cout << thread_rank << " owlqn l" << std::endl;
    if(local_thread_id == main_thread_id){
        for(int j = 0; j < train_data->fea_dim; j++){
            *(all_nodes_global_g + j) = 0.0;
        }
        for(int j = 0; j < train_data->fea_dim; j++){//must be pay attention
            *(global_g + j) /= threads_num;
        }
        std::cout << thread_rank << " owlqn m" << std::endl;
        MPI_Allreduce(global_g, all_nodes_global_g, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);//all_nodes_global_g store shared sum of every nodes search direction
    }
    std::cout << thread_rank << " owlqn n" << std::endl;
    pthread_barrier_wait(&barrier);
    //should be synchronous all threads
    std::cout << thread_rank << " owlqn o" << std::endl;
    line_search(all_nodes_global_g);//use global search direction to search
    //update slist
    std::cout << thread_rank << " owlqn p" << std::endl;
    if(local_thread_id == main_thread_id){
        cblas_daxpy(train_data->fea_dim, -1, (double*)w, 1, (double*)next_w, 1);
        cblas_dcopy(train_data->fea_dim, (double*)next_w, 1, (double*)s_list[(m - use_list_len) % m], 1);
    //update ylist
        cblas_daxpy(train_data->fea_dim, -1, (double*)global_g, 1, (double*)global_next_g, 1);
        cblas_dcopy(train_data->fea_dim, (double*)global_next_g, 1, (double*)y_list[(m - use_list_len) % m], 1);

        use_list_len++;
        if(use_list_len > m){
            for(int j = 0; j < train_data->fea_dim; j++){
                *(*(s_list + abs(m - use_list_len) % m) + j) = 0.0;
                *(*(y_list + abs(m - use_list_len) % m) + j) = 0.0;
            }
        }
        cblas_dcopy(train_data->fea_dim, (double*)next_w, 1, (double*)w, 1);
    }
    pthread_barrier_wait(&barrier);
    std::cout << thread_rank << " owlqn q" << std::endl;
}

void LR::owlqn(int proc_id, int n_procs){
    std::cout << train_data->fea_dim << std::endl;
    std::cout << thread_rank << " owlqn start" << std::endl;
    float *ro_list = new float[train_data->fea_dim];

    std::cout << thread_rank << " owlqn a" << std::endl;
    float **s_list = new float*[m];
    s_list[0] = new float[m * train_data->fea_dim];
    for(int i = 1; i < m; i++){
        s_list[i] = s_list[i-1] + train_data->fea_dim;
    }

    std::cout << thread_rank << " owlqn b" << std::endl;
    float **y_list = new float* [m];
    y_list[0] = new float[m * train_data->fea_dim];
    for(int i = 1; i < m; i++){
        y_list[i] = y_list[i-1] + train_data->fea_dim;
    }

    std::cout << thread_rank << " owlqn c" << std::endl;
    int use_list_len = 0;
    int step = 0;
    while(step < 3){
        std::cout << thread_rank << " owlqn d" << std::endl;
        parallel_owlqn(use_list_len, ro_list, s_list, y_list);
        step++;
    }
    std::cout << thread_rank << " owlqn r" << std::endl;
    //free memory
    delete [] ro_list;
    for(int i = 0; i < m; i++){
        delete [] s_list[i];
        delete [] y_list[i];
    }
    delete s_list;
    delete y_list;
    std::cout << thread_rank << " owlqn s" << std::endl;
}
