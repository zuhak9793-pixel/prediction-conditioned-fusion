// Paper5 compatibility/interface layer; upstream mathematics in c_NIMF.cpp.
#pragma once
#include <cstring>
#include "c_NIMF.h"
class Core {
public:
    int n,m,d; double lambda,alpha,eta,last;
    double **R,**U,**S,**gU,**gS;
    vector<vector<pair<int,double>>> W,rev;
    Core(double* r,double* u,double* s,int nn,int mm,int dd,double l,double a,double e):
        n(nn),m(mm),d(dd),lambda(l),alpha(a),eta(e) {
        R=vector2Matrix(r,n,m);U=vector2Matrix(u,n,d);S=vector2Matrix(s,m,d);
        gU=createMatrix(n,d);gS=createMatrix(m,d);rev.resize(n);
        W=getPCCMatrix(R,n,m,10,rev);
        last=loss(U,S,R,lambda,alpha,n,m,d,W);
    }
    ~Core(){deleteMatrix(gU);deleteMatrix(gS);delete[] R;delete[] U;delete[] S;}
    double objective(){return loss(U,S,R,lambda,alpha,n,m,d,W);}
    void gradient(double* gu,double* gs){
        gradLoss(U,S,R,gU,gS,lambda,alpha,n,m,d,W,rev);
        memcpy(gu,gU[0],n*d*sizeof(double));memcpy(gs,gS[0],m*d*sizeof(double));
    }
    double step(){
        gradLoss(U,S,R,gU,gS,lambda,alpha,n,m,d,W,rev);
        double rate=linesearch(U,S,R,last,gU,gS,eta,lambda,alpha,n,m,d,W);
        for(int k=0;k<d;k++){
            for(int i=0;i<n;i++)U[i][k]-=rate*gU[i][k];
            for(int j=0;j<m;j++)S[j][k]-=rate*gS[j][k];
        }
        last=objective();return rate;
    }
    double at(int i,int j){
        double v=alpha*dotProduct(U[i],S[j],d);
        for(auto w:W[i])v+=(1-alpha)*w.second*dotProduct(U[w.first],S[j],d);
        return v;
    }
    void coords(const long long* rows,const long long* cols,int count,double* out){
        for(int k=0;k<count;k++)out[k]=at((int)rows[k],(int)cols[k]);
    }
    void weights(double* out){for(int i=0;i<n;i++)for(auto w:W[i])out[i*n+w.first]=w.second;}
};
