#include <iostream>

namespace dmlc{
namespace linear{

  class Scheduler : public ps::App{
    public:
        Scheduler(){}
        ~Scheduler(){}
	virtual bool Run(){
	    std::cout<<"Connected "<<ps::NodeInfo::NumServers()<<" servers and "<<ps::NodeInfo::NumWorkers()<<" workers"<<std::endl;
	}
  };

}//end linear
}//end dmlc
