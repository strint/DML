#include "lr.h"
#include "load_data.h"
#include "mpi.h"
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include <iomanip>
#include "gtest/gtest.h"
#include "config.h"

#ifndef MPI_MAX_PROCESSOR_NAME
#define MPI_MAX_PROCESSOR_NAME 100
#endif
struct ThreadParam{
    LR *lr;
    int threads_num;
    int process_id;
    int n_process;
};

void *opt_algo(void *arg){
   ThreadParam *args = (ThreadParam*)arg;
   args->lr->owlqn(args->process_id, args->n_process); 
}

int main(int argc,char* argv[]){  
    std::cout<<"cmd: ./train -trainfile -testfile"<<std::endl;
    int rank, numprocs, namelen=100, rank_num = 3, tag = 1;
    MPI_Request req = {};
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    MPI_Status status;
    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&numprocs);
    MPI_Get_processor_name(processor_name, &namelen);
    //exec by main thread    
    const char *train_data_file = argv[1];
    char *test_data_file = argv[2];
    std::string split_tag = " ";
    
    Load_Data load_data;
    load_data.fea_dim = 0;
    load_data.load_data(train_data_file, split_tag);
    int root = 0;
    //MPI_Bcast(&load_data.fea_dim, 1, MPI_INT, root, MPI_COMM_WORLD);
    if(rank != 0){
        //MPI_Send(&send_fea_dim, 1, MPI_UINT64_T, 0, tag, MPI_COMM_WORLD, &req);
        MPI_Send(&load_data.fea_dim, 1, MPI_UINT64_T, 0, tag, MPI_COMM_WORLD);
    }
    else if(rank == 0){
        sleep(3);
        int64_t sum = 0;
        int64_t recv_data;
        while(rank_num--){
            //MPI_Recv(&recv_data, 1, MPI_UINT64_T, MPI_ANY_SOURCE, tag, MPI_COMM_WORLD, &req);
            MPI_Recv(&recv_data, 1, MPI_UINT64_T, MPI_ANY_SOURCE, tag, MPI_COMM_WORLD, &status);
            sum += recv_data;
        }
        std::cout<<sum<<std::endl;
    }
    std::vector<ThreadParam> params;
    std::vector<pthread_t> threads;
    CONFIG config;
    config.n_threads = 2;
    LR lr;
    lr.rank = rank;
    lr.feature_dim = load_data.fea_dim;
    lr.init_theta();
    //lr.data = static_cast<void*>(&load_data);
    lr.data = &load_data;
    for(int i = 0; i < config.n_threads; i++){//construct parameter
        ThreadParam param = {&lr, config.n_threads, rank, numprocs};
        params.push_back(param);
    } 
    //multithread start
    for(int i = 0; i < params.size(); i++){
        pthread_t thread_id;
        //std::cout<<thread_id<<std::endl;
        int ret = pthread_create(&thread_id, NULL, &opt_algo, (void*)&(params[i])); 
        if(ret != 0) std::cout<<"process "<<i<<"failed(create thread faild.)"<<std::endl;
        else threads.push_back(thread_id);
    }
    for(int i = 0; i < threads.size(); i++){//join threads function
        pthread_join(threads[i], 0); 
    }
   
    MPI::Finalize();
    return 0;
}
