#ifndef FTRL_H
#define FTRL_H
#include "load_data.h"

class FTRL{

public:
	FTRL(Load_Data* data, int total_num_proc, int my_rank);
	~FTRL();
	float* glo_w;

private:
	Load_Data* data;
	void init();
	float sigmoid(float x);
  	int step;
	float* loc_w;
	float* loc_g;
	float* glo_g;

	float* loc_z;
	float* loc_sigma;
	float* loc_n;

	float alpha;
	float beta;
	float lambda1;
	float lambda2;

	int num_proc;
	int rank;
};

#endif
