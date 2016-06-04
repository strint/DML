#ifndef FTRL_H
#define FTRL_H
#include "load_data.h"
#include "mpi.h"
#include <math.h>

class FTRL{

public:
	FTRL(Load_Data* data, int total_num_proc, int my_rank);
	~FTRL();
	float* glo_w;
	float* loc_w;
        void run();
private:
	Load_Data* data;
	void init();
	float sigmoid(float x);
	void update_w();
        void update_other_parameter();
	void ftrl();
  	int step;

	float* loc_g;
	float* glo_g;

	float* loc_z;
	float* loc_sigma;
	float* loc_n;

	float alpha;
	float beta;
	float lambda1;
	float lambda2;
	int batch_size;

	int num_proc;
	int rank;
};

#endif
