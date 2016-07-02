#ifndef FTRL_H
#define FTRL_H
#include "load_data.h"
#include "ps/ps.h"
#include <math.h>

class FTRL{

public:
	sigmoid(float x) {
	    if(x < -30) return 1e-6;
    	    else if(x > 30) return 1.0;
            else{
                double ex = pow(2.718281828, x);
                return ex / (1.0 + ex);
            }
	}

	FTRL(Load_Data& batch_data) : data(batch_data){
	    pctr_.resize(data->label.size());
            keys.clear();
            x = 0.0;
            for(int i = 0; i < data.size(); i++){
		wx = bias;
	 	kv_->Wait(kv_->Pull(keys, &w));
                for(int j = 0; j < data[i]; j++){
                    key = data[i][j].idx;
                    x = data[i][j].val;
                    keys.push_back(key);
		    wx += w[key] * x; 
                    X_.push_back(value);
                    XX_.push_back(value*vaule);
                }
		calcgrad();
	    }
 	}
	~FTRL();
        calcgrad(){
	    float pctr = sigmoid(wx);  
	    
	}
	} 
        void run();
private:
	Load_Data* data;
	ps::KVWorker<float> * kv_;
	std::vector<long int> keys;
	float bias = 0.0;
	float wx;
	long int key;
	float x;
	std::vector<float> w;
	std::vector<float> v;
  	std::vector<float> pctr_;
        std::vector<float> X_, XX_;

        int k;
};

#endif
