#ifndef LOAD_DATA_H_
#define LOAD_DATA_H_

#include <iostream>
#include <vector>
#include "data_struct.h"

class LoadData{
public:
    LoadData();
    ~LoadData();
    void load_data(char *train_file, std::string split_tag);
    void get_feature_struct();
    std::vector<std::string> split_line(std::string split_tag, std::vector<std::string>& feature_index);
    std::vector<std::vector<sparse_feature> > fea_matrix;
    std::vector<float> label; 
};
#endif
