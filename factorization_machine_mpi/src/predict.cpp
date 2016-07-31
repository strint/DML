#include <iostream>
#include <fstream>
#include "predict.h"

Predict::Predict(Load_Data* ld, int total_num_proc, int my_rank) : data(ld), num_proc(total_num_proc), rank(my_rank){

}
Predict::~Predict(){}

void Predict::predict(std::vector<float> glo_w){
    int y = 0.0;
    std::vector<float> predict_result;
    for(int i = 0; i < data->loc_ins_num; i++) {
	int x = 0.0;
        for(int j = 0; j < data->fea_matrix[i].size(); j++) {
            int idx = data->fea_matrix[i][j].idx;
            int val = data->fea_matrix[i][j].val;
            x += glo_w[idx] * val;
        }
        if(x < -30){
            y = 1e-6;
        }
        else if(x > 30){
            y = 1.0;
        }
        else{
            double ex = pow(2.718281828, x);
            y = ex / (1.0 + ex);
        }
        predict_result.push_back(y);
    }
    for(size_t j = 0; j < predict_result.size(); j++){
        if(rank == 0){
	     std::cout<<predict_result[j]<<"\t"<<1 - data->label[j]<<"\t"<<data->label[j]<<"\t"<<rank<<std::endl;
	}
    }
}

