#include "ftrl.h"
#include <stdlib.h>

void FTRL::run(){
    for(int i = 0; i < step; i++){
        std::cout<<"step "<<i<<std::endl;
        data.load_data_minibatch(1000);
        calcgrad();
	    float wx = bias;
	    for(int col = 0; col < data->fea_matrix[row].size(); col++){//for one instance
	  	index = data->fea_matrix[row][col].idx;
	        value = data->fea_matrix[row][col].val;
	        wx += loc_w[index] * value;
	    }
            for(int k = 0; k < factor; k++){
		float vxvx = 0.0, vvxx = 0.0;
		for(int col = 0; col < data->fea_matrix[row].size(); col++){
		    index = data->fea_matrix[row][col].idx;
                    value = data->fea_matrix[row][col].val;
                    vxvx += loc_v_arr[col][k] * value;
		    vvxx += loc_v_arr[col][k] * loc_v_arr[col][k] * value*value;
                }
	        vxvx *= vxvx;
		vxvx -= vvxx;
		wx += vxvx * 1.0 / 2;	
            }
	    pctr = sigmoid(wx);
            //std::cout<<"pctr "<<pctr<<std::endl;
	    loss_sum = (pctr - data->label[row]);
            //std::cout<<"loss_sum "<<loss_sum<<std::endl;
	    for(int l = 0; l < data->glo_fea_dim; l++){
		//std::cout<<"l = "<<l<<"\tglo_fea_dim ="<<data->glo_fea_dim<<std::endl;
	        loc_g_w[l] += loss_sum * value;
	        float vx = 0;
 		for(int k = 0; k < factor; k++){
		    for(int j = 0; j != l && j < data->glo_fea_dim; j++){
			if(loc_v_arr[j][k] == 0.0) continue;
			vx +=  loc_v_arr[j][k] * value;
		    }
                    loc_g_v_arr[l][k] += loss_sum * vx;
		}	
	    }
//std::cout<<row<<std::endl;
	    ++row;
    	}
        update_other_parameter();     
    }//end for
}

