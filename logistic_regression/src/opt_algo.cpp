#include "opt_algo.h"
#include "mpi.h"
#include <iostream>
#include <vector>

extern "C"{
#include <cblas.h>
}

OPT_ALGO::OPT_ALGO(){
}

OPT_ALGO::~OPT_ALGO(){
    delete [] w; 
    delete [] next_w;
    delete [] global_g;
    delete [] global_next_g;
    delete [] all_nodes_global_g;
}

std::vector<std::string> OPT_ALGO::split_line(std::string split_tag, std::vector<std::string>& feature_index){
    int start = 0, end = 0;
    while((end = line.find_first_of(split_tag, start)) != std::string::npos){
        if(end > start){
            index_str = line.substr(start, end - start);
            feature_index.push_back(index_str);
        }
        start = end + 1;
    }
    if(start < line.size()){
        index_str = line.substr(start);
        feature_index.push_back(index_str);
    }
}

void OPT_ALGO::get_feature_struct(){
    for(int i = 1; i < feature_index.size(); i++){//start from index 1
	int start = 0, end = 0;
	while((end = feature_index[i].find_first_of(":", start)) != std::string::npos){
            if(end > start){
	        index_str = feature_index[i].substr(start, end - start);
	        index = atoi(index_str.c_str());
                if(index > fea_dim) fea_dim = index + 1;
		sf.idx = index - 1;
            }
            //beg += 1; //this code must remain,it makes me crazy two days!!!
            start = end + 1;
	}
	if(start < feature_index[i].size()){
            index_str = feature_index[i].substr(start);
            value = atoi(index_str.c_str());
            sf.val = value;
        }
        key_val.push_back(sf);
    }
}

void OPT_ALGO::load_data(std::string data_file, std::string split_tag){
    std::ifstream fin(data_file.c_str(), std::ios::in);
    if(!fin) std::cerr<<"open error get feature number..."<<data_file<<std::endl;
    int y = 0;
    while(getline(fin,line)){
        feature_index.clear();
        key_val.clear();
        //return id:value, .e.g 3:1, 4:1
        split_line(split_tag, feature_index);
        y = atof(feature_index[0].c_str());
        label.push_back(y);
        //3:1 as input
        get_feature_struct();
        fea_matrix.push_back(key_val);
    }
    fin.close();
}

void OPT_ALGO::init_theta(){
    c = 1.0;
    m = 10;
    n_threads = 3;

    w = new float[fea_dim];
    next_w = new float[fea_dim];
    global_g = new float[fea_dim];
    global_next_g = new float[fea_dim];
    all_nodes_global_g = new float[fea_dim];

    global_old_loss_val = 0.0;
    global_new_loss_val = 0.0;
    
    main_thread_id = getpid();   

    sync_global_g = 3;
    sync_s_y_list = 3;
    sync_global_old_loss = 3;
    sync_global_new_loss = 3;

 
    float init_w = 0.0;
    for(int j = 0; j < fea_dim; j++){
        *(w + j) = init_w;
        *(next_w + j) = init_w;
        *(global_g + j) = init_w;
        *(global_next_g + j) = init_w; 
    }
}

//----------------------------owlqn--------------------------------------------
float OPT_ALGO::sigmoid(float x)
{
    float sgm = 1/(1+exp(-(float)x));
    return (float)sgm;
}

float OPT_ALGO::loss_function_value(float *para_w){
    float f = 0.0;
    for(int i = 0; i < fea_matrix.size(); i++){
        float x = 0.0;
        for(int j = 0; j < fea_matrix[i].size(); j++){
            int id = fea_matrix[i][j].idx;
            float val = fea_matrix[i][j].val;
            x += *(para_w + id) * val;//maybe add bias later
        }
        float l = label[i] * log(1/sigmoid(-1 * x)) + (1 - label[i]) * log(1/sigmoid(x));
        f += l;
    }
    return f;
}

void OPT_ALGO::loss_function_gradient(float *para_w, float *para_g){
    float f = 0.0;
    for(int i = 0; i < fea_matrix.size(); i++){
        float x = 0.0, value = 0.0;
        int index;
        for(int j = 0; j < fea_matrix[i].size(); j++){
            index = fea_matrix[i][j].idx;
            value = fea_matrix[i][j].val;
            x += *(para_w + index) * value;
        }
        for(int j = 0; j < fea_matrix[i].size(); j++){
            *(para_g + j) += label[i] * sigmoid(x) * value + (1 - label[i]) * sigmoid(x) * value;
        }
    }
    for(int j = 0; j < fea_dim; j++){
        *(para_g + j) /= fea_matrix.size();
    }
}

void OPT_ALGO::loss_function_subgradient(float * local_g, float *local_sub_g){
    if(c == 0.0){
        for(int j = 0; j < fea_dim; j++){
            *(local_sub_g + j) = -1 * *(local_g + j);
        }
    }
    else{
        for(int j = 0; j < fea_dim; j++){
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

void OPT_ALGO::fix_dir(float *w, float *next_w){
    for(int j = 0; j < fea_dim; j++){
        if(*(next_w + j) * *(w + j) >=0) *(next_w + j) = 0.0;
        else *(next_w + j) = *(next_w + j);
    }
}

void OPT_ALGO::line_search(float *param_g){
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
        local_thread_id = getpid();
        if(local_thread_id == main_thread_id){
            MPI_Allreduce(&global_old_loss_val, &all_nodes_old_loss_val, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
            sync_global_old_loss--;
        }
        else{
            sync_global_old_loss--;
        }
        while(sync_global_old_loss > 0){
            sleep(1);
        }
        for(int j = 0; j < fea_dim; j++){
            *(next_w + j) = *(w + j) + alpha * *(param_g + j);//local_g equal all nodes g
        }
        fix_dir(w, next_w);//orthant limited
        new_loss_val = loss_function_value(next_w);//cal new loss per thread

        pthread_mutex_lock(&mutex);
        global_new_loss_val += new_loss_val;//sum all threads loss value
        pthread_mutex_unlock(&mutex);

        if(local_thread_id == main_thread_id){
            MPI_Allreduce(&global_new_loss_val, &all_nodes_new_loss_val, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);//sum all nodes loss.
            sync_global_new_loss--;
        }
        else{
            sync_global_new_loss--;
        }
        while(sync_global_new_loss > 0){
            sleep(1);
        }
        loss_function_gradient(next_w, global_next_g);

        if(all_nodes_new_loss_val <= all_nodes_old_loss_val + beta * cblas_ddot(fea_dim, (double*)param_g, 1, (double*)global_next_g, 1)){
            break;
        }
        alpha *= backoff;
        break;
    }
}

void OPT_ALGO::two_loop(int use_list_len, float *local_sub_g, float **s_list, float **y_list, float *ro_list, float *p){
    float *q = new float[fea_dim];//local variable
    float *alpha = new float[m]; 
    cblas_dcopy(fea_dim, (double*)local_sub_g, 1, (double*)q, 1);
    if(use_list_len < m) m = use_list_len; 

    for(int loop = 1; loop <= m; ++loop){
        ro_list[loop - 1] = cblas_ddot(fea_dim, (double*)(&(*y_list)[loop - 1]), 1, (double*)(&(*s_list)[loop - 1]), 1);
        alpha[loop] = cblas_ddot(fea_dim, (double*)(&(*s_list)[loop - 1]), 1, (double*)q, 1)/ro_list[loop - 1];
        cblas_daxpy(fea_dim, -1 * alpha[loop], (double*)(&(*y_list)[loop - 1]), 1, (double*)q, 1);
    }
    delete [] q;
    float *last_y = new float[fea_dim];
    for(int j = 0; j < fea_dim; j++){
        last_y[j] = *((*y_list + m - 1) + j);
    }

    float ydoty = cblas_ddot(fea_dim, (double*)last_y, 1, (double*)last_y, 1);
    float gamma = ro_list[m - 1]/ydoty;
    cblas_sscal(fea_dim, gamma,(float*)p, 1);

    for(int loop = m; loop >=1; --loop){
        float beta = cblas_ddot(fea_dim, (double*)(&(*y_list)[m - loop]), 1, (double*)p, 1)/ro_list[m - loop];
        cblas_daxpy(fea_dim, alpha[loop] - beta, (double*)(&(*s_list)[m - loop]), 1, (double*)p, 1);
    }
    delete [] alpha;
    delete [] last_y;
}

void OPT_ALGO::parallel_owlqn(int use_list_len, float* ro_list, float** s_list, float** y_list){
    //define and initial local parameters
    float *local_g = new float[fea_dim];//single thread gradient
    float *local_sub_g = new float[fea_dim];//single thread subgradient
    float *p = new float[fea_dim];//single thread search direction.after two loop
    loss_function_gradient(w, local_g);//calculate gradient of loss by global w)
    loss_function_subgradient(local_g, local_sub_g); 
    //should add code update multithread and all nodes sub_g to global_sub_g
    two_loop(use_list_len, local_sub_g, s_list, y_list, ro_list, p);

    pthread_mutex_lock(&mutex);
    for(int j = 0; j < fea_dim; j++){
        *(global_g + j) += *(p + j);//update global direction of all threads
    }
    pthread_mutex_unlock(&mutex);

    pid_t local_thread_id;
    local_thread_id = getpid();
    if(local_thread_id == main_thread_id){
        for(int j = 0; j < fea_dim; j++){ 
            *(all_nodes_global_g + j) = 0.0;
        }
        for(int j = 0; j < fea_dim; j++){//must be pay attention
            *(global_g + j) /= n_threads;
        }   
        MPI_Allreduce(global_g, all_nodes_global_g, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);//all_nodes_global_g store shared sum of every nodes search direction
        sync_global_g--;
    }
    else{
        sync_global_g--;
    }
    while(sync_global_g > 0){
        sleep(1);
    }
    //should be synchronous all threads
    line_search(all_nodes_global_g);//use global search direction to search
    //update slist
    if(local_thread_id == main_thread_id){
        cblas_daxpy(fea_dim, -1, (double*)w, 1, (double*)next_w, 1);
        cblas_dcopy(fea_dim, (double*)next_w, 1, (double*)s_list[(m - use_list_len) % m], 1);
    //update ylist
        cblas_daxpy(fea_dim, -1, (double*)global_g, 1, (double*)global_next_g, 1); 
        cblas_dcopy(fea_dim, (double*)global_next_g, 1, (double*)y_list[(m - use_list_len) % m], 1);

        use_list_len++;
        if(use_list_len > m){
            for(int j = 0; j < fea_dim; j++){
                *(*(s_list + abs(m - use_list_len) % m) + j) = 0.0;
                *(*(y_list + abs(m - use_list_len) % m) + j) = 0.0;        
            }
        }
        cblas_dcopy(fea_dim, (double*)next_w, 1, (double*)w, 1);
        sync_s_y_list--;
    }
    else{
        sync_s_y_list--;
    }
    while(sync_s_y_list > 0){
        sleep(1);
    }
}

void OPT_ALGO::owlqn(int proc_id, int n_procs){
    float *ro_list = new float[fea_dim];

    float **s_list = new float*[m];
    s_list[0] = new float[m * fea_dim];
    for(int i = 1; i < m; i++){
        s_list[i] = s_list[i-1] + fea_dim; 
    }
    
    float **y_list = new float* [m];
    y_list[0] = new float[m * fea_dim];
    for(int i = 1; i < m; i++){
        y_list[i] = y_list[i-1] + fea_dim; 
    }

    int use_list_len = 0;
    int step = 0;
    while(step < 3){
        parallel_owlqn(use_list_len, ro_list, s_list, y_list);        
        step++;
    }
    //free memory
    delete [] ro_list;
    for(int i = 0; i < m; i++){
        delete [] s_list[i];
        delete [] y_list[i];
    }
    delete s_list;
    delete y_list;
}

