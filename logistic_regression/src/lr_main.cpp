#include "lr.h"
#include "load_data.h"
#include "mpi.h"
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include <iomanip>
#include "gtest/gtest.h"
#include "config.h"

struct ThreadParam{
    int threads_num;
    int thread_rank;
    int process_id;
    int n_process;
    Load_Data *train_data;
};

void *opt_algo(void *arg){
    ThreadParam *args = (ThreadParam*)arg;
    std::cout << "I'm thread " << args->thread_rank << " of total " << args->threads_num << " threads" << std::endl;
    LR lr;
    lr.threads_num = args->threads_num;
    lr.thread_rank = args->thread_rank;
    lr.train_data = args->train_data;
    lr.init_theta();
    lr.owlqn(args->thread_rank, args->threads_num);
}


int main(int argc,char* argv[]){
    std::cout<<"cmd: ./train -trainfile -testfile"<<std::endl;
    int myid, numprocs;
    MPI_Status status;
    MPI_Init(NULL,NULL);
    MPI_Comm_rank(MPI_COMM_WORLD,&myid);
    MPI_Comm_size(MPI_COMM_WORLD,&numprocs);

    //exec by main thread
    std::cout << "Num of process: " << numprocs << std::endl;
    const char *train_data_file = argv[1];
    char *test_data_file = argv[2];
    std::string split_tag = " ";

    Load_Data train_data;
    train_data.fea_dim = 0;
    train_data.load_data(train_data_file, split_tag);

    int root = 0;
    //MPI_Bcast(&ld.fea_dim, 1, MPI_INT, root, MPI_COMM_WORLD);
    std::cout << "Trainning data dimension: " << train_data.fea_dim << std::endl;
    std::cout << "Trainning data num: " << train_data.get_data_num() << std::endl;
    std::vector<ThreadParam> params;
    std::vector<pthread_t> threads;
    //CONFIG config;
    int n_threads = 2;
    for(int i = 0; i < n_threads; i++){//construct parameter
        ThreadParam param = {n_threads, i, n_threads, i, &train_data};
        params.push_back(param);
    }
    //multithread start
    for(int i = 0; i < params.size(); i++){
        pthread_t thread_id;
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
