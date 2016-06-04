#include <iostream>
#include <fstream>
#include <string>
#include <cmath>
#include <glog/logging.h>
#include "load_data.h"
#include "stdlib.h"

Load_Data::Load_Data(){}
Load_Data::~Load_Data(){}

void Load_Data::load_data(const char* data_file, std::string split_tag, int rank, int nproc){
    MPI_Status status;
    std::ifstream fin(data_file, std::ios::in);
    LOG(INFO) << "process " << rank << " read "<< data_file << std::endl;
    if(!fin.is_open()) {
        LOG(ERROR) << "process "<< rank << " open file error: " << data_file << std::endl;
        exit(1);
    }

    while(!fin.eof()){
        std::getline(fin, line);
	key_val.clear();
	const char *pline = line.c_str();
	if(sscanf(pline, "%f%n", &y, &nchar) >= 1){
	    pline += nchar;
	    label.push_back(y);
	    while(sscanf(pline, "%d:%f%n", &index, &value, &nchar) >= 2){
		pline += nchar;
		sf.idx = index;
	        if(sf.idx + 1 > loc_fea_dim) loc_fea_dim = sf.idx + 1;
		sf.val = value;
	        key_val.push_back(sf); 
	    }
	}
        fea_matrix.push_back(key_val);
        loc_ins_num++;
    }
    fin.close();

    if(rank != MASTER_ID){
        //LOG(INFO) << "process " << rank <<" send loc_fea_dim" << std::endl;
        MPI_Send(&loc_fea_dim, 1, MPI_INT, MASTER_ID, FEA_DIM_FLAG, MPI_COMM_WORLD);
    } else {
	    if(loc_fea_dim > glo_fea_dim) glo_fea_dim = loc_fea_dim;
	    for(int i = 1; i < nproc; i++){
            //LOG(INFO) << "process " << rank <<" revc process "<< i << std::endl;
            long int other_loc_fea_dim;
	        MPI_Recv(&other_loc_fea_dim, 1, MPI_INT, i, FEA_DIM_FLAG, MPI_COMM_WORLD, &status);
            //LOG(INFO) << "process "<< rank <<" revc process " << i << " over" << std::endl;
	        if(other_loc_fea_dim > glo_fea_dim) glo_fea_dim = loc_fea_dim;
	    }
    }
    //LOG(INFO) << "process "<< rank << " glo_fea_dim " << glo_fea_dim << " before Bcast" << std::endl;
    MPI_Bcast(&glo_fea_dim, 1, MPI_INT, MASTER_ID, MPI_COMM_WORLD);
    //LOG(INFO) << "process "<< rank << " glo_fea_dim " << glo_fea_dim << " after Bcast" << std::endl;
}
