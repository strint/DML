#ifndef FTRL_H
#define FTRL_H
#include "load_data.h"
#include <math.h>
#include "ps/ps.h"

struct FTRLEntry{
    float w = 0.0;
    float sq_cum_grad = 0.0;
    float z = 0.0;
};

class FTRL{

public:
	FTRL(Load_Data* data);
	~FTRL();
        void run();
private:
	Load_Data* data;
	ps::KVWorker<float>* kv_; 
	void init();
	float sigmoid(float x);
	void updateW(std::vector<long int>&, std::vector<float>&, std::vector<float>&);
	float alpha;
	float beta;
	float lambda1;
	float lambda2;
	int step;
	float bias;
};

#endif
