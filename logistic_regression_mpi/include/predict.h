#ifndef PREDICT_H_
#define PREDICT_H_

#include <string>
#include <vector>
#include <math.h>
#include "load_data.h"

class Predict{

public:
    Predict(Load_Data* data, int total_num_proc, int my_rank);
    ~Predict();

    void predict(std::vector<float>);
    Load_Data* data;
    float pctr;

    //MPI process info
    int num_proc; // total num of process in MPI comm world
    int rank; // my process rank in MPT comm world 

};
#endif
