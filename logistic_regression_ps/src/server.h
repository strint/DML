#include "iostream"
#include "ps.h"

namespace dmlc{
namespace linear{
    struct ISGDHandle{
      public:
        ISGDHandle(){ ns_ = ps::NodeInfo::NumServers();}
        float alpha = 0.1, beta = 1.0;        
      private:
        int ns_ = 0;
        static int64_t new_w;
    };  
    struct FTRLEntry{
        float w = 0;
        float z = 0;
        float sq_cum_grad = 0;
    };
    struct FTRLHandle : public ISGDHandle{
    public:
        inline void Push(ps::Key key, ps::Blob<const float> grad, FTRLEntry& val){
	    float g = grad[0];
            float sqrt_n = val.sq_cum_grad;
            float sqrt_n_new = sqrt(sqrt_n * sqrt_n + g * g);
                val.z += g - (sqrt_n_new - sqrt_n);
                val.sq_cum_grad = sqrt_n_new;
                float z = val.z;
                if(abs(z) <= lambda1){
                    val.w = 0.0;
                }
                else{
                    float tmpr= 0.0;
                    if(z >= 0) tmpr = z - lambda1;
                    else tmpr = z + lambda1;
                    float tmpl = -1 * ( ( beta + val.sq_cum_grad - sqrt_n) / alpha  + lambda2);
                    val.w = tmpr / tmpl;
                }
        }
        int lambda1 = 1.0;
        int lambda2 = 1.0;
    };
    template <typename Entry, typename Handle>
    class SERVER : public ps::App{
    public:
        SERVER(){
            CreateServer<FTRLEntry, FTRLHandle>();
        }
        void CreateServer(){
            Handle h;
            ps::OnlineServer<float, Entry, Handle> s(h);
        }
        ~SERVER(){}
    };
}//end linear
}//end dmlc
