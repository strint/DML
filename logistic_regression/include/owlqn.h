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
    LR(Load_Data* data);
    ~LR();
    void run(int nproc, int rank);

private:
    Load_Data* data;

    double c;
    int m;
    double lambda;

    double *w;
    double *next_w;//model paramter after line search
    double *global_w; 

    double *g;//gradient of loss function
    double *sub_g;
    //double *next_g;
    double *q;

    double loss;
    double new_loss;

    double **s_list;
    double **y_list;
    double *alpha;
    double *ro_list;
    

    void init();
    void owlqn(int rank, int n_proc);
    void calculate_gradient();
    void calculate_subgradient();
    void two_loop();
    void line_search();
    double calculate_loss(double *w);
    double sigmoid(double x);
    void fix_dir();
};
#endif // LR_H_

