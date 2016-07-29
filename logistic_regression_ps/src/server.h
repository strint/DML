#include "iostream"

namespace dmlc{
namespace linear{

    struct ISGDHandle{
        public:
        ISGDHandle(){ ns_ = ps::NodeInfo::NumServers()}
       
        float alpha = 0.1, beta = 1.0;        
 
        private:
        int ns_ = 0;
    };  
  
    struct FTRLEntry{
        float w = 0;
        float z = 0;
        float sq_cum_grad = 0;
    };

    struct FTRLHandle : public ISGDHandle{
    public:
        inline void Push(ps::Key key, Blob<const float> grad, FTRLEntry& val){
            float sqrt_n = entrys[i].sq_cum_grad;
                float sqrt_n_new = sqrt(sqrt_n * sqrt_n + g[i] * g[i]);
                entrys[i].z += g[i] - (sqrt_n_new - sqrt_n);
                entrys[i].sq_cum_grad = sqrt_n_new;
                float z = entrys[i].z;
                if(abs(z) <= lambda1){
                    entrys[i].w = 0.0;
                }
                else{
                    float tmpr= 0.0;
                    if(z >= 0) tmpr = z - lambda1;
                    else tmpr = z + lambda1;
                    float tmpl = -1 * ( ( beta + entrys[i].sq_cum_grad - sqrt_n) / alpha  + lambda2);
                    entrys[i].w = tmpr / tmpl;
                }
        }
    };

    class SERVER : ps::App{
    public:
        SERVER(){
            CreateServer<FTRLEntry, FTRLHandle>();
        }
        
        void CreateServer(){
            ps::OnlineServer<float, Entry, Handle> s(h);
        }
        ~SERVER(){}
    };
}//end linear
}//end dmlc
