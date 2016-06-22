#include <string>
#include "load_data.h"
#include "owlqn.h"
#include "ftrl.h"
#include "mpi.h"
#include "predict.h"
//#include "gtest/gtest.h"
#include <glog/logging.h>
#include "config.h"

int main(int argc,char* argv[]){  
    int rank, nproc;
    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&nproc);
    // Initialize Google's logging library.
    google::InitGoogleLogging(argv[0]);
    FLAGS_log_dir = "./log";
    LOG(INFO) << "my process rank: "<< rank <<", totoal process num: "<< nproc <<std::endl;

    char train_data_path[1024];
    const char *train_data_file = argv[2];
    snprintf(train_data_path, 1024, "%s-%05d", train_data_file, rank);
    char test_data_path[1024];
    const char *test_data_file = argv[3];
    snprintf(test_data_path, 1024, "%s-%05d", test_data_file, rank);

    std::string split_tag = " ";
    Load_Data load_data; 
    load_data.load_data(train_data_path, split_tag, rank, nproc);
    //std::cout<<ld.fea_matrix.size()<<std::endl;    
    std::vector<float> model;
    if (strcmp(argv[1], "owlqn")){
        OWLQN owlqn(&load_data, nproc, rank);
        owlqn.run();
        for(int j = 0; j < load_data.glo_fea_dim; j++){
            //std::cout<<"w["<< j << "]: "<<ftrl.loc_w[j]<<std::endl;
            model.push_back(owlqn.glo_w[j]);
        }
    }
    else if(strcmp(argv[1], "ftrl")){
        FTRL ftrl(&load_data, nproc, rank);
        ftrl.run();
        for(int j = 0; j < load_data.glo_fea_dim; j++){
	    //std::cout<<"w["<< j << "]: "<<ftrl.loc_w[j]<<std::endl;
	    model.push_back(ftrl.loc_w[j]);
        }
    }

    load_data.load_data(test_data_path, split_tag, rank, nproc);
    Predict predict(&load_data, nproc, rank);
    predict.predict(model);
   
    MPI::Finalize();
    return 0;
}
