#include <iostream>
#include <vector>
#include "data_struct.h"

class LoadData{
public:
    LoadData();
    ~LoadData();
    void loan_data(char *train_file, std::string split_tag);
    std::vector<std::vector<sparse_feature> > fea_matrix;
    std::vector<float> label; 
};
