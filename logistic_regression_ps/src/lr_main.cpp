#include "server.h"
#include "ps.h"

namespace ps{
  App* App::Create(int argc, char *argv[]){
    int rank = ps::MyRank();
    char train_data_path[1024];
    const char *train_data_file = argv[1];
    snprintf(train_data_path, 1024, "%s-%05d", train_data_file, rank);

    NodeInfo n;
    if(n.IsWorker()){
        return new ::dmlc::linear::Worker(train_data_path);
    }else if(n.IsServer()){
        return new ::dmlc::linear::Server();
    }else if(n.IsScheduler){
        return new ::dmlc::linear::Scheuler();
    }
    return NULL;
  }
}//namespace ps

int64_t dmlc::linear::ISGDHandle::new_w = 0;

int main(int argc,char *argv[]){  
    return ps::RunSystem(&argc, &argv);
}
