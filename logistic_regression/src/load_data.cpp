#include <iostream>
#include <fstream>
#include <string>
#include <cmath>
#include "load_data.h"
#include "stdlib.h"

Load_Data::Load_Data(){}
Load_Data::~Load_Data(){}

void Load_Data::split_line(const std::string& line, const std::string& split_tag, std::vector<std::string>& feature_index) {
    size_t start = 0, end = 0;
    feature_index.clear();
    while((end = line.find_first_of(split_tag, start)) != std::string::npos) {
        if(end > start){
            feature_index.push_back(line.substr(start, end - start));
        }
        start = end + 1;
    }
    if(start < line.size()){
        feature_index.push_back(line.substr(start));
    }
}

void Load_Data::get_feature_struct(std::vector<std::string>& feature_index, std::vector<sparse_feature>& key_val){
    key_val.clear();
    sparse_feature sf;
    std::string index_str;
    std::string value_str;
    for(int i = 1; i < feature_index.size(); i++){//start from index 1
        int start = 0, end = 0;
        while((end = feature_index[i].find_first_of(":", start)) != std::string::npos){
            if(end > start){
                index_str = feature_index[i].substr(start, end - start);
                sf.idx = atoi(index_str.c_str());
                if(sf.idx + 1 > loc_fea_dim) loc_fea_dim = sf.idx + 1;
            }
            //beg += 1; //this code must remain,it makes me crazy two days!!!
            start = end + 1;
        }
        if(start < feature_index[i].size()){
            value_str = feature_index[i].substr(start);
            sf.val = atof(value_str.c_str());
        }
        key_val.push_back(sf);
    }
}

void Load_Data::load_data(const char* data_file, std::string split_tag, int rank, int nproc){
    MPI_Status status;
    std::ifstream fin(data_file, std::ios::in);
    data_rank = rank;
    if(!fin) std::cerr << "process "<< rank << " open file error: " << data_file << std::endl;
    //std::cout << "read "<< data_file << std::endl;
    double y = 0.0;
    std::string line;
    std::vector<std::string> feature_index;
    std::vector<sparse_feature> key_val;
    while(getline(fin, line)){
        split_line(line, split_tag, feature_index);
        y = atof(feature_index[0].c_str());
        label.push_back(y);
        get_feature_struct(feature_index, key_val);
        fea_matrix.push_back(key_val);
        loc_samp_num++;
    }
    fin.close();

    if(rank != MASTER_ID){
        //std::cout << "process " << rank <<" send loc_fea_dim" << std::endl;
        MPI_Send(&loc_fea_dim, 1, MPI_INT, MASTER_ID, FEA_DIM_FLAG, MPI_COMM_WORLD);
    } else {
	    if(loc_fea_dim > glo_fea_dim) glo_fea_dim = loc_fea_dim;
	    for(int i = 1; i < nproc; i++){
            //std::cout << "process " << rank <<" revc process "<< i << std::endl;
            long int other_loc_fea_dim;
	        MPI_Recv(&other_loc_fea_dim, 1, MPI_INT, i, FEA_DIM_FLAG, MPI_COMM_WORLD, &status);
            //std::cout << "process "<< rank <<" revc process " << i << " over" << std::endl;
	        if(other_loc_fea_dim > glo_fea_dim) glo_fea_dim = loc_fea_dim;
	    }
    }
    //std::cout << "process "<< rank << " glo_fea_dim " << glo_fea_dim << " before Bcast" << std::endl;
    MPI_Bcast(&glo_fea_dim, 1, MPI_INT, MASTER_ID, MPI_COMM_WORLD);
    //std::cout << "process "<< rank << " glo_fea_dim " << glo_fea_dim << " after Bcast" << std::endl;
}
