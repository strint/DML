#ifndef LR_H_
#define LR_H_

#include <string>
#include <fstream>
#include <vector>
#include <math.h>
#include <stdlib.h>
#include <iostream>
#include <string.h>
#include <deque>
#include <pthread.h>
#include "load_data.h"

class LR{
public:
    LR();
    ~LR();
    //call by main thread
    void init_theta();
    //call by threads 
    void owlqn(int proc_id, int n_procs);
    //shared by multithreads
    float *w;//model paramter shared by all threads
    float *next_w;//model paramter after line search
    float *global_g;//gradient of loss function
    float *global_next_g;//gradient of loss function when arrive new w
    float *all_nodes_global_g;
    float global_old_loss_val;//loss value of loss function
    float all_nodes_old_loss_val;
    float global_new_loss_val;//loss value of loss function when arrive new w
    float all_nodes_new_loss_val;
    //void* data;
    Load_Data* data;
    pid_t main_thread_id;
    int feature_dim;
    float c;
    int m;
    int rank;
//private:
    void parallel_owlqn(int use_list_len, float* ro_list, float** s_list, float** y_list);
    void loss_function_gradient(float *para_w, float *para_g);
    void loss_function_subgradient(float *local_g, float *local_sub_g);
    void two_loop(int use_list_len, float *sub_g, float **s_list, float **y_list, float *ro_list, float *p);
    void line_search(float *local_g);
    float loss_function_value(float *w);
    float sigmoid(float x);
    void fix_dir(float *w, float *next_w);

    pthread_mutex_t mutex;
    pthread_barrier_t barrier;
};
#endif
