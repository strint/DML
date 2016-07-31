#include <string>
#include "load_data.h"
#include "ps.h"

namespace ps{
  App* App::Create(int argc, char *argv[]){
    int rank = ps::MyRank();
    char train_data_path[1024];
    const char *train_data_file = argv[1];
    snprintf(train_data_path, 1024, "%s-%05d", train_data_file, rank);

    NodeInfo n;
    if(n.IsWorker()){
        return new WORKER(train_data_path);
    }else if(n.IsServer()){
        return new SERVER();
    }else if(n.IsScheduler){
        return new SCHEDULER();
    }
    return NULL;
  }
}

int64_t dmlc::linear::ISGDHandle::new_w = 0;

int main(int argc,char *argv[]){  
    retrun ps::RunSystem(&argc, &argv);
}
