#!/usr/bin/env python
# coding: utf-8

# In[18]:


import triangle
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import dsolve
import scipy.sparse as sparse
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


# ## Plot de mesh

# In[1]:


def barycentres(v,t):
    n=len(t)
    b=np.zeros((n,2))
    for i in range(n):
        b[i]=sum([v[j]*1./3 for j in t[i]])
    return b

#args: (mesh,*option)
#Options: 
# 'v' : afficher les sommets
# 't' : afficher les triangles
# 'V' : vertex markers, 1 si exterieur, 0 si interieur, 2 si bord obstacle

def plotMesh(*args):
    mesh=args[0]  
    v=mesh['vertices']
    t=mesh['triangles']
    x=np.array([s[0] for s in v])
    y=np.array([s[1] for s in v])
    plt.triplot(x,y,t)
    
    if len(args)>=2: 
        option=args[1]
        v1=v+np.array([0.01,0.01])
        if (option.find('v')) != -1:
            for i in range(len(v)):
                plt.text(v1[i][0],v1[i][1],str(i+1))
        if (option.find('t')) != -1:
            b=barycentres(v,t)
            for i in range(len(t)):
                plt.text(b[i][0],b[i][1],str(i+1),color='red')
        if (option.find('V')) != -1:
            label=mesh['vertex_markers']
            for i in range(len(v)):
                plt.text(v1[i][0],v1[i][1],str(label[i][0]))
        if (option.find('d')) != -1:
            if len(args)<3:
                print("erreur: ef absent de l'input")
                return -1
            ef=args[2]
            ddl=ef['ddl']
            numDdl=ef['numDdl']
            for i in range(len(ddl)):
                plt.text(ddl[i][0],ddl[i][1],str(i+1))   
    plt.show()


# ## Integration 2D, 1D

# In[20]:


def formInt2D(numero):
    if numero== 1:
        return (1./3,1./3,1./2)
    if numero== 2:
        return (np.array([1./2,1./2,0]),np.array([1./2,0,1./2]),np.array([1./6]*3))
    if numero== 3:
        return (np.array([1./6,1./6,2./3]),np.array([1./6,2./3,1./6]),np.array([1./6]*3))
    if numero== 4:
        return (np.array([1./3,1./5,3./5,1./5]),np.array([1./3,1./5,1./5,3./5]),np.array([-9./32]+[25./96]*3))
    if numero== 5:
        return (np.array([1./2,1./2,0,1./6,1./6,2./3]),np.array([1./2,0,1./2,1./6,2./3,1./6]),np.array([1./60]*3+[3./20]*3))
    if numero== 6:
        a=(6+np.sqrt(15))/21
        b=(6-np.sqrt(15))/21
        A=(155+np.sqrt(15))/2400
        B=(155-np.sqrt(15))/2400
        return (np.array([1./3,a,1-2*a,a,b,1-2*b,b]),np.array([1./3,a,a,1-2*a,b,b,1-2*b]),np.array([9./80]+[A]*3+[B]*3))
    print("formInt2D: numero inconnu")
    return -1

def formInt1D(numero):
    if numero== 1:
        return 0.5
    if numero== 2:
        return (np.array([1./2-1./(2*np.sqrt(3)),1./2+1./(2*np.sqrt(3))]),np.array([1./2,1./2]))
    if numero== 3:
        return (np.array([0.5-0.5*np.sqrt(3./5),0.5,0.5+0.5*np.sqrt(3./5)]),np.array([5./18]*3))
    print("formInt1D: numero inconnu")
    return -1

def calc_det_integ(vect):
    B_A=np.dot(np.array([-1,1,0]),vect)
    C_A=np.dot(np.array([-1,0,1]),vect)
    tmp=np.dot(C_A,np.array([[0,1],[-1,0]]))
    return np.abs(np.dot(tmp*B_A,np.array([1,1])))

def buildInteg(mesh,numero):
    (xhat,yhat,what)=formInt2D(numero)
    tlist=mesh['triangles']
    vlist=mesh['vertices']
    t=vlist[tlist]
    size=len(t)*len(xhat)
    integ=dict(x=np.zeros(size),y=np.zeros(size),w=np.zeros(size)) 
    for i in range(len(xhat)):
        slr=np.array([1-yhat[i]-xhat[i],yhat[i],xhat[i]])
        V=np.dot(slr,t)
        (integ['x'][i::len(xhat)],integ['y'][i::len(xhat)])=np.transpose(V)
        integ['w'][i::len(xhat)]=what[i]*calc_det_integ(t)
    return integ


# ## Éléments finis de Lagrange

# In[2]:


def f(x):
    p=x[0]
    return np.exp(p)


#utiliser .transpose() pour s'accorder à l'input
def lagrange2D(x,y,ordre):
    x=np.transpose(x)
    y=np.transpose(y)
    if ordre==2:
        return ([np.transpose([1+2*x*x-3*x+2*y*y-3*y+4*x*y, 2*x*x-x, 2*y*y-y, -4*x*(x-1)-4*x*y, 4*x*y, -4*y*(y-1)-4*x*y]),
                 np.transpose([4*x-3+4*y, 4*x-1, x-x, -8*x+4-4*y, 4*y, -4*y]),
                 np.transpose([4*y-3+4*x, x-x, 4*y-1, -4*x, 4*x, -8*y+4-4*x])])
    if ordre==1:
        return ([np.transpose([1-x-y,x,y]),np.transpose([-1+x-x,1+x-x,x-x]),np.transpose([-1+x-x,x-x,1+x-x])])
    if ordre==0:
        return ([np.transpose([1-x+x]),np.transpose([x-x]),np.transpose([x-x])])
    return -1

def get_ddl_ordre2(mesh):
    v=mesh['vertices']
    e=mesh['edges']
    mrk=mesh['vertex_markers'] 
    b=np.zeros((len(v)+len(e),2))
    ef_mrk=np.zeros(len(v)+len(e))
    for i in range(len(v)):
        b[i]=v[i]
        ef_mrk[i]=mrk[i]
    for j in range(len(e)):
        b[i+j+1]=(v[e[j][0]]+v[e[j][1]])*1./2
        if ef_mrk[e[j][0]]+ef_mrk[e[j][1]]==2:
            ef_mrk[i+j+1]=1
    return (b,ef_mrk)

def get_numDdl_ordre2(mesh):
    t=mesh['triangles']
    e=mesh['edges']
    v=mesh['vertices']
    b=np.zeros((len(t),6))
    Ntri=len(t)
    L_t=range(Ntri)
    i=0
    for i in range(len(t)):
        b[i][:3]=np.array(t[i])
        b[i][3]=len(v)+np.where(np.logical_or(np.less(np.dot(np.abs(e-np.array([t[i][0],t[i][1]])),np.ones(2)),0.00001),np.less(np.dot(np.abs(e-np.array([t[i][1],t[i][0]])),np.ones(2)),0.00001)))[0][0]
        b[i][5]=len(v)+np.where(np.logical_or(np.less(np.dot(np.abs(e-np.array([t[i][0],t[i][2]])),np.ones(2)),0.00001),np.less(np.dot(np.abs(e-np.array([t[i][2],t[i][0]])),np.ones(2)),0.00001)))[0][0]
        b[i][4]=len(v)+np.where(np.logical_or(np.less(np.dot(np.abs(e-np.array([t[i][1],t[i][2]])),np.ones(2)),0.00001),np.less(np.dot(np.abs(e-np.array([t[i][2],t[i][1]])),np.ones(2)),0.00001)))[0][0]
    return b

"""
Input formule: pour la formule d'integration

Structure de l'output ef: 
    - ef['ddl'][i] : coordonnées du i-ème degrés de liberté
    - ef['numDdl'][i][j] : numéro du j-ème ddl dans le i-ème triangle
    - ef['ddl markers'][i] : vaut 1 si le i-ème ddl est sur le bord et 0 sinon
    - ef['u'][i][j] : évaluation de la j-ème fonction de base au i-ème point d'integration
    - ef['dxu'][i][j] et ef['dyu'][i][j] : idem pour pour les dérivées en x et y des fonctions de base
"""
def FEspace(mesh, typeEF, ordre, formule):
    v=mesh['vertices']
    t=mesh['triangles']
    Ntri=len(t)
    
    ef=dict()
    if ordre==0:
        ef['ddl']=np.array(print_ef.barycentres(v,t))
        ef['numDdl']=np.array([[i] for i in range(Ntri)])
        ef['ddl markers']=np.zeros(len(ef['ddl']))
    if ordre==1:
        ef['ddl']=np.array(v)
        ef['numDdl']=np.array(t)
        ef['ddl markers']=mesh['vertex_markers']
    if ordre==2:
        (ef['ddl'],ef['ddl markers'])=get_ddl_ordre2(mesh)
        ef['numDdl']=get_numDdl_ordre2(mesh)
    Nddl=len(ef['ddl'])
    (xhat,yhat,omega)=formInt2D(formule)    
    (phi,dx_phi,dy_phi)=lagrange2D(xhat,yhat,ordre)
    Nint=len(xhat)
    Nb=len(phi[0])
    M = sparse.lil_matrix((Nint*Ntri,Nddl))
    vrt=np.transpose(ef['numDdl'])
    for iloc in range(Nint):
        for jloc in range(Nb):
            M[iloc+Nint*np.arange(Ntri),vrt[jloc]]=phi[iloc][jloc]
    M=M.tocsc()
    ef['u']=M

    
    B_A=np.dot(np.array([-1,1,0]),v[t])
    C_A=np.dot(np.array([-1,0,1]),v[t])
    dx_x=np.transpose(B_A)[0]
    dy_x=np.transpose(B_A)[1]
    dx_y=np.transpose(C_A)[0]
    dy_y=np.transpose(C_A)[1]
    tmp=np.dot(C_A,np.array([[0,1],[1,0]]))
    det=np.dot(np.multiply(B_A,tmp),np.array([1,-1]))
    idx_x=dy_y*1./det
    idy_x=-dx_y*1./det
    idx_y=-dy_x*1./det
    idy_y=dx_x*1./det    
    Mx = sparse.lil_matrix((Nint*Ntri,Nddl))
    My = sparse.lil_matrix((Nint*Ntri,Nddl))
    for iloc in range(Nint):
        for jloc in range(Nb):
            Mx[iloc+Nint*np.arange(Ntri),vrt[jloc]]=dx_phi[iloc][jloc]*idx_x+dy_phi[iloc][jloc]*idx_y
            My[iloc+Nint*np.arange(Ntri),vrt[jloc]]=dx_phi[iloc][jloc]*idy_x+dy_phi[iloc][jloc]*idy_y
    Mx=Mx.tocsc()
    My=My.tocsc()
    ef['dxu']=Mx
    ef['dyu']=My
    
    return ef


# ## Integration CL Neumann

# In[22]:


#forme bilinéaire
def integrate_UV_N(ef,integ):
    A_1=ef['u'].T*sparse.diags(integ['w'])*ef['u']
    A_2=ef['dxu'].transpose()*sparse.diags(integ['w'])*ef['dxu']
    A_3=ef['dyu'].transpose()*sparse.diags(integ['w'])*ef['dyu']
    return A_1+A_2+A_3

#forme linéaire
def integrate_fV_N(f,ef,integ):
    F=sparse.diags(integ['w'])*f
    return ef['u'].T*F

def L2_norm_compare(U,Uh,ef,integ):
    diff=U-Uh
    A=ef['u'].T*sparse.diags(integ['w'])*ef['u']
    return np.sqrt((diff.T).dot(A*diff))

def H1_norm_compare(U,Uh,ef,integ):
    diff=U-Uh
    A=integrate_UV(ef,integ)
    return np.sqrt((diff.T).dot(A*diff))


# ## Integration CL Dirichlet homogène

# In[23]:


#Matrice P de transition vers les sommets intérieurs à utiliser pour l'intégration
def get_HDir_mat(ef,mesh):
    idx=np.where(ef['ddl markers']==0)[0]
    P = sparse.lil_matrix((len(idx),len(ef['ddl markers'])))
    P[:len(idx),idx]=sparse.diags(np.ones(len(idx)))
    return (P.T,idx)

def L2_norm_compare_Dir(U,Uh,ef,integ,P):
    diff=U-Uh
    A=ef['u'].T*sparse.diags(integ['w'])*ef['u']
    At=P.T*A*P
    return np.sqrt((diff.T).dot(At.dot(diff)))

def H1_norm_compare_Dir(U,Uh,ef,integ,P):
    diff=U-Uh
    At=P.T*A*P
    return np.sqrt((diff.T).dot(At.dot(diff)))

