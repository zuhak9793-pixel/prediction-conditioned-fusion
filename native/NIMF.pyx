# cython: language_level=3
# Paper5 minimal binding; upstream Jamie Zhu / WS-DREAM, MIT (see LICENSE).
import numpy as np
cimport numpy as np
np.import_array()
cdef extern from "bridge.h":
    cdef cppclass Core:
        Core(double*,double*,double*,int,int,int,double,double,double) except +
        double objective() except +
        double step() except +
        void gradient(double*,double*) except +
        void coords(const long long*,const long long*,int,double*) except +
        void weights(double*) except +

cdef class Model:
    cdef Core* core
    cdef object r,u,s
    def __cinit__(self, R, U, S, double lmda=30, double alpha=.4, double eta=.01):
        self.core=NULL
        R=np.array(R,dtype=np.float64,order='C',copy=True)
        U=np.array(U,dtype=np.float64,order='C',copy=True)
        S=np.array(S,dtype=np.float64,order='C',copy=True)
        if R.ndim!=2 or U.ndim!=2 or S.ndim!=2: raise ValueError('2D arrays required')
        if min(R.shape+U.shape+S.shape)<=0: raise ValueError('empty dimensions')
        if U.shape[0]!=R.shape[0] or S.shape[0]!=R.shape[1] or U.shape[1]!=S.shape[1]:raise ValueError('shape mismatch')
        if not all(np.isfinite(x).all() for x in [R,U,S]) or (R<0).any():raise ValueError('finite positive training / zero missing required')
        if not np.isfinite([lmda,alpha,eta]).all() or lmda<0 or not 0<=alpha<=1 or eta<=0:raise ValueError('invalid parameters')
        self.r=R;self.u=U;self.s=S
        self.core=new Core(<double*>(<np.ndarray>R).data,<double*>(<np.ndarray>U).data,<double*>(<np.ndarray>S).data,R.shape[0],R.shape[1],U.shape[1],lmda,alpha,eta)
    def __dealloc__(self):
        if self.core!=NULL:del self.core
    def objective(self):return self.core.objective()
    def step(self):return self.core.step()
    def factors(self):return self.u.copy(),self.s.copy()
    def gradient(self):
        cdef np.ndarray gu=np.empty_like(self.u),gs=np.empty_like(self.s)
        self.core.gradient(<double*>gu.data,<double*>gs.data)
        return gu,gs
    def weights(self):
        cdef np.ndarray w=np.zeros((self.r.shape[0],self.r.shape[0]),dtype=np.float64)
        self.core.weights(<double*>w.data)
        return w
    def predict(self, rows, cols):
        if np.asarray(rows).dtype.kind not in 'iu' or np.asarray(cols).dtype.kind not in 'iu':raise ValueError('integer coordinates required')
        cdef np.ndarray rr=np.array(rows,dtype=np.int64,order='C',copy=True)
        cdef np.ndarray cc=np.array(cols,dtype=np.int64,order='C',copy=True)
        if rr.ndim!=1 or cc.ndim!=1 or len(rr)!=len(cc):raise ValueError('1D aligned coordinates required')
        if len(rr)>2147483647 or (rr<0).any() or (rr>=self.r.shape[0]).any() or (cc<0).any() or (cc>=self.r.shape[1]).any():raise ValueError('coordinate bounds')
        cdef np.ndarray out=np.empty(len(rr),dtype=np.float64)
        self.core.coords(<long long*>rr.data,<long long*>cc.data,len(rr),<double*>out.data)
        return out
