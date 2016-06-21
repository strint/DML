#ifndef LOAD_DATA_H_
#define LOAD_DATA_H_

#include <iostream>
#include <vector>
#include "data_struct.h"
#include "mpi.h"

#define MASTER_ID (0)
#define FEA_DIM_FLAG (99)

class Load_Data {
public:
    Load_Data();
    ~Load_Data();
    void load_data(const char *train_file, std::string split_tag, int rank, int nproc);
    std::string line;
    int y, value;
    int nchar, index;
    std::vector<std::vector<sparse_feature> > fea_matrix;
    std::vector<sparse_feature> key_val;
    sparse_feature sf;
    std::vector<int> label;
    long int loc_fea_dim = 0;
    long int glo_fea_dim = 0;
    long int loc_ins_num = 0;

private:
};
#endif
