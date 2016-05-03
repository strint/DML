#include "owlqn.h"
#include "mpi.h"
#include <string.h>
//#include "gtest/gtest.h"
#include "config.h"

int main(int argc,char* argv[]){  
    int rank, nproc;

    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&nproc);
    std::cout<<"my process rank: "<<rank<<", totoal process num: "<<nproc<<std::endl;
    const char *train_data_file = argv[1];
    const char *test_data_file = argv[2];
    std::string split_tag = " ";
    Load_Data ld; 
    ld.load_data(train_data_file, split_tag, rank, nproc);
    std::cout<< "process "<< rank <<", local feature dimension: "<<ld.loc_fea_dim<<std::endl;
    LR lr(&ld, nproc, rank);
    //lr.run();
    MPI::Finalize();
    return 0;
}
