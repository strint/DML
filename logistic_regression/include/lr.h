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
    double *w;//model paramter shared by all threads
    double *next_w;//model paramter after line search
    double *global_g;//gradient of loss function
    double *global_next_g;//gradient of loss function when arrive new w
    double *all_nodes_global_g;
    double global_old_loss_val;//loss value of loss function
    double all_nodes_old_loss_val;
    double global_new_loss_val;//loss value of loss function when arrive new w
    double all_nodes_new_loss_val;
    //void* data;
    Load_Data* data;
    pid_t main_thread_id;
    int feature_dim;
    double c;
    int m;
    int rank;
//private:
    void parallel_owlqn(int use_list_len, double* ro_list, double** s_list, double** y_list);
    void loss_function_gradient(double *para_w, double *para_g);
    void loss_function_subgradient(double *local_g, double *local_sub_g);
    void two_loop(int use_list_len, double *sub_g, double **s_list, double **y_list, double *ro_list, float *p);
    void line_search(double *local_g);
    double loss_function_value(double *w);
    double sigmoid(double x);
    void fix_dir(double *w, double *next_w);

    pthread_mutex_t mutex;
    pthread_barrier_t barrier;
};
#endif
