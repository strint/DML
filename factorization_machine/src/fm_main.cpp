#include <string>
#include "load_data.h"
#include "predict.h"
#include "ftrl.h"
#include "mpi.h"
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
    const char *train_data_file = argv[1];
    snprintf(train_data_path, 1024, "%s-%05d", train_data_file, rank);
    char test_data_path[1024];
    const char *test_data_file = argv[2];
    snprintf(test_data_path, 1024, "%s-%05d", test_data_file, rank);

    std::string split_tag = " ";
    Load_Data ld; 
    ld.load_data(train_data_path, split_tag, rank, nproc);
    //std::cout<<ld.fea_matrix.size()<<std::endl;    
    FTRL ftrl(&ld, nproc, rank);
    ftrl.run();
    std::vector<float> model;
    for(int j = 0; j < ld.glo_fea_dim; j++){
	//std::cout<<"w["<< j << "]: "<<ftrl.loc_w[j]<<std::endl;
	model.push_back(ftrl.loc_w[j]);
    }
    Load_Data testdata;
    testdata.load_data(test_data_path, split_tag, rank, nproc);
    Predict p(&testdata, nproc, rank);
    //p.predict(model);
   
    MPI::Finalize();
    return 0;
}
