#ifndef OPT_ALGO_H_
#define OPT_ALGO_H_

#include <string>
#include <fstream>
#include <vector>
#include <math.h>
#include <stdlib.h>
#include <iostream>
#include <string.h>
#include <deque>
#include <pthread.h>
#include<unistd.h>

struct sparse_feature{
    int idx;
    float val;
};

class OPT_ALGO{
public:
    OPT_ALGO();
    ~OPT_ALGO();

    //call by main thread
    void load_data(std::string train_data_file, std::string split_tag);
    void init_theta();

    //call by threads 
    void owlqn(int proc_id, int n_procs);
    //shared by multithreads
    std::vector<std::vector<sparse_feature> > fea_matrix;//feature matrix shared by all threads
    std::vector<float> label;//label of instance shared by all threads
    float *w;//model paramter shared by all threads
    float *next_w;//model paramter after line search
    float *global_g;//gradient of loss function
    float *all_nodes_global_g;
    float *global_next_g;//gradient of loss function when arrive new w
    long int fea_dim;//feature dimension
    float c;
    int m;
    int n_threads;//thread number
    float global_old_loss_val;//loss value of loss function
    float all_nodes_old_loss_val;
    float global_new_loss_val;//loss value of loss function when arrive new w
    float all_nodes_new_loss_val;
    pid_t main_thread_id;

private:
    std::vector<std::string> split_line(std::string split_tag, std::vector<std::string>& feature_index); 
    void get_feature_struct();
    void parallel_owlqn(int use_list_len, float* ro_list, float** s_list, float** y_list);
    void loss_function_gradient(float *para_w, float *para_g);
    void loss_function_subgradient(float *local_g, float *local_sub_g);
    void two_loop(int use_list_len, float *sub_g, float **s_list, float **y_list, float *ro_list, float *p);
    void line_search(float *local_g);
    float loss_function_value(float *w);
    float sigmoid(float x);
    void fix_dir(float *w, float *next_w);

    pthread_mutex_t mutex;
    int sync_global_g;
    int sync_s_y_list;
    int sync_global_old_loss;
    int sync_global_new_loss;

    std::string line;
    std::vector<std::string> tmp_vec; 
    std::vector<std::string> feature_index;
    std::string index_str;
    sparse_feature sf;
    std::vector<sparse_feature> key_val;
    int index, value;
};
#endif
