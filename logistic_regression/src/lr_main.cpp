#include <string>
#include "load_data.h"
#include "owlqn.h"
#include "mpi.h"
//#include "gtest/gtest.h"
#include "config.h"

int main(int argc,char* argv[]){  
    int rank, nproc;

    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&nproc);
    std::cout<<"my process rank: "<<rank<<", totoal process num: "<<nproc<<std::endl;

    char train_data_path[1024];
    const char *train_data_file = argv[1];
    snprintf(train_data_path, 1024, "%s-%05d", train_data_file, rank);
    char test_data_path[1024];
    const char *test_data_file = argv[2];
    snprintf(test_data_path, 1024, "%s-%05d", test_data_file, rank);

    std::string split_tag = " ";
    Load_Data ld; 
    ld.load_data(train_data_path, split_tag, rank, nproc);
    std::cout<<"process "<<rank<<" sample number: "<<ld.loc_samp_num<<std::endl;

    LR lr(&ld, nproc, rank);
    lr.run();

    MPI::Finalize();
    return 0;
}
