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
//#include <unistd.h>
#include "load_data.h"

class LR{

public:
    LR(Load_Data* data);
    ~LR();
    void run(int nproc, int rank);
    double *global_w; 

private:
    Load_Data* data;
    double *w;
    double *next_w;//model paramter after line search
    double *g;//gradient of loss function
    double *sub_g;
    //double *next_g;
    double old_loss;//loss value of loss function
    double new_loss;//loss value of loss function when arrive new w
    double c;
    int m;
    double lambda;
    
    double** s_list;
    double** y_list;
    double *alpha;
    double *ro_list；
    double* q;
    
    double loss;
    double new_loss;

    void init_theta();
    void owlqn(int rank, int n_proc);
    void calculate_gradient();
    void calculate_subgradient();
    void two_loop();
    void line_search();
    double calculate_loss(double *w);
    double sigmoid(double x);
    void fix_dir();
};
#endif
