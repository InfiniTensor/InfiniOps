__device__ float pair_accumulate_exact(float p0,float p1,float c) {
    const double a=double(p0),b=double(p1),cc=double(c);
    const double s=__dadd_rn(a,b),t=__dadd_rn(s,cc);
    if(!isfinite(t))return __double2float_rn(t);
    // Products of finite FP16 values are exact FP32 values. Recover both
    // FP64 addition residuals, including exact cancellation and midpoint ties.
    const double vb=__dsub_rn(s,a);
    const double e=__dadd_rn(__dsub_rn(a,__dsub_rn(s,vb)),__dsub_rn(b,vb));
    const double vc=__dsub_rn(t,s);
    const double f=__dadd_rn(__dsub_rn(s,__dsub_rn(t,vc)),__dsub_rn(cc,vc));
    const double error=__dadd_rn(e,f);
    const float r=__double2float_rn(t);
    if(error==0.)return r;
    if(r==0.f)return __double2float_rn(error);
    const unsigned bits=__float_as_uint(r);
    const float upper=__uint_as_float(r>0.f?bits+1:bits-1);
    const float lower=__uint_as_float(r>0.f?bits-1:bits+1);
    const double hm=__dmul_rn(__dadd_rn(double(r),double(upper)),.5);
    const double lm=__dmul_rn(__dadd_rn(double(r),double(lower)),.5);
    const double hd=__dadd_rn(__dsub_rn(t,hm),error);
    const double ld=__dadd_rn(__dsub_rn(t,lm),error);
    if(hd>0. || (hd==0. && (bits&1)))return upper;
    if(ld<0. || (ld==0. && (bits&1)))return lower;
    return r;
}
