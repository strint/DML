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
    void run(int nproc, int rank);
    double *global_w; 

private:
    double *w;
    double *next_w;//model paramter after line search
    double *g;//gradient of loss function
    double *next_g;
    double old_loss;//loss value of loss function
    double new_loss;//loss value of loss function when arrive new w
    double c;
    int m;
    double lambda;

    void init_theta();
    void owlqn(int proc_id, int n_procs);
    void parallel_owlqn(int step, int use_list_len, double* ro_list, double** s_list, double** y_list, int rank, int nproc);
    void calculate_gradient(double *w, double *g);
    void calculate_subgradient(double *g, double *sub_g);
    void two_loop(int step, int use_list_len, double *sub_g, double **s_list, double **y_list, double *ro_list, float *p);
    void line_search(double *local_g);
    double loss_function_value(double *w);
    double sigmoid(double x);
    void fix_dir(double *w, double *next_w);
};
#endif
