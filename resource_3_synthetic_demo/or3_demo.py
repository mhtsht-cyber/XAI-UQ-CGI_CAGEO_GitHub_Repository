# Repository Resource 3 for Computers & Geosciences
# Article: Explainable and Uncertainty-Aware Computational Geodetic Inversion: A Review and Framework for Multi-Source Earth Observation
# Authors: Mohit Sheode and D. Kishan
# Corresponding author: Mohit Sheode, Department of Civil Engineering, MANIT Bhopal, India; mhtsht@gmail.com; 223111001@stu.manit.ac.in

"""
Repository Resource 3 (v3.0) -- reproducibility script for Section 8.3
Controlled same-model synthetic proof-of-concept of the XAI-UQ-CGI pipeline.

Forward model: Mogi point pressure source in a homogeneous, isotropic elastic
half-space (flat surface, static snapshot). Data: synthetic InSAR LOS (single
descending look; planar atmospheric ramp estimated jointly as nuisance terms)
and sparse 3-component GNSS. Engines: weighted TRF least squares + affine-
invariant ensemble MCMC (four OVERDISPERSED, independently seeded ensembles).
Convergence via rank-normalized split-R-hat and bulk-ESS (ArviZ). Includes
per-parameter sensor ablation, a 300-realization calibration of the LINEARIZED
least-squares intervals, and one deliberate model-mismatch experiment.

Run:  python or3_demo.py
Generates: f6..f11 PNGs, insar_obs.csv, gnss_obs.csv, config.json,
           decision_record.json, ablation.json, diagnostics.json,
           requirements_lock.txt
"""
import json, platform, csv, numpy as np, matplotlib, matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from scipy.stats import norm
import emcee, corner, arviz as az, scipy

RNG = np.random.default_rng(20260729)
NU  = 0.25

# ---------------------------------------------------------------- config
CFG = dict(
    seed=20260729, nu=NU,
    m_true=dict(x0=1000.0, y0=-500.0, d=3500.0, dV=3.0e6),
    grid=dict(xmin=-4000, xmax=4000, ymin=-4000, ymax=4000, n=41,
              spacing_m=200, keep_prob=0.25, decimation="fixed Bernoulli mask, fixed seed"),
    los_enu_raw=[0.621, -0.112, 0.776],
    los_sign="positive LOS = motion toward satellite (range decrease)",
    ramp_true=dict(c0=2.0e-3, cx=1.2e-6, cy=-0.8e-6,
                   treatment="estimated jointly as 3 nuisance parameters (InSAR block only)"),
    noise=dict(insar=0.015, gnss_h=0.002, gnss_v=0.005, dist="Gaussian iid zero-mean"),
    gnss_xy=[[-3000,-2500],[-1500,2500],[0,-3200],[500,1500],[2800,-1000],
             [-2800,800],[1800,3000],[3200,2200],[-500,-800]],
    param_names=["x0","y0","d","dV","c0","cx","cy"],
    param_units=["m","m","m","m^3","m","m/m","m/m"],
    prior="uniform over [lower, upper] bounds",
    bounds_lower=[-4000,-4000, 300, 1e4, -0.05, -1e-5, -1e-5],
    bounds_upper=[ 4000, 4000, 9000, 2e7,  0.05,  1e-5,  1e-5],
    ls_initial=[0.0, 0.0, 2000.0, 1.0e6, 0.0, 0.0, 0.0],
    mcmc_init_scales=[200,200,300,2e5,2e-3,2e-7,2e-7],
    solver=dict(pkg="scipy.optimize.least_squares", method="trf", loss="linear",
                x_scale="jac", ftol=1e-12, xtol=1e-12, jac="2-point finite difference",
                covariance="(J^T J)^-1 scaled by reduced chi-square (local linearized)"),
    mcmc=dict(nwalkers=40, nsteps=9000, burn=2250, thin=5, n_ensembles=4,
              dispersion=4.0, dispersion_note="centers ~ N(LS, (4*init_scale)^2), clipped to bounds"),
    mc=dict(n_realizations=300, interval="linearized least-squares normal intervals"),
)
LB = np.array(CFG["bounds_lower"]); UB = np.array(CFG["bounds_upper"])
MT = np.array([CFG["m_true"][k] for k in ("x0","y0","d","dV")])
LOS = np.array(CFG["los_enu_raw"]); LOS = LOS/np.linalg.norm(LOS)
CFG["los_enu_normalized"] = [round(v,4) for v in LOS.tolist()]
RT = np.array([CFG["ramp_true"][k] for k in ("c0","cx","cy")])
PNAMES = ["x0 (m)","y0 (m)","d (m)","dV (m^3)","c0 (m)","cx","cy"]

# ---------------------------------------------------------------- forward
def mogi_enu(src, x, y):
    x0,y0,d,dV = src
    dx,dy = x-x0, y-y0
    R = np.sqrt(dx*dx+dy*dy+d*d)
    c = (1.0-NU)/np.pi*dV/R**3
    return c*dx, c*dy, c*d
def insar_los(src, ramp, x, y):
    ue,un,uu = mogi_enu(src, x, y)
    return ue*LOS[0]+un*LOS[1]+uu*LOS[2] + ramp[0]+ramp[1]*x+ramp[2]*y

# ---------------------------------------------------------------- geometry
g = CFG["grid"]
gx = np.linspace(g["xmin"],g["xmax"],g["n"]); gy = np.linspace(g["ymin"],g["ymax"],g["n"])
GX,GY = np.meshgrid(gx,gy)
mask = RNG.random(GX.size) < g["keep_prob"]
XI,YI = GX.ravel()[mask], GY.ravel()[mask]
N_INSAR = int(mask.sum())
XG = np.array([p[0] for p in CFG["gnss_xy"]],float)
YG = np.array([p[1] for p in CFG["gnss_xy"]],float)
N_GNSS = XG.size
CFG["n_insar"]=N_INSAR; CFG["n_gnss"]=N_GNSS
S_I,S_H,S_V = CFG["noise"]["insar"],CFG["noise"]["gnss_h"],CFG["noise"]["gnss_v"]

def make_data(rng, src=MT, second=None):
    los = insar_los(src, RT, XI, YI); ge,gn,gu = mogi_enu(src, XG, YG)
    if second is not None:
        l2 = insar_los(second,[0,0,0],XI,YI); e2,n2,u2 = mogi_enu(second,XG,YG)
        los=los+l2; ge,gn,gu = ge+e2,gn+n2,gu+u2
    los = los + rng.normal(0,S_I,los.size)
    ge = ge+rng.normal(0,S_H,N_GNSS); gn = gn+rng.normal(0,S_H,N_GNSS); gu = gu+rng.normal(0,S_V,N_GNSS)
    return los,ge,gn,gu
LOS_OBS,GE_O,GN_O,GU_O = make_data(RNG)

# export observations (deterministic, from this run)
with open('insar_obs.csv','w',newline='') as f:
    w=csv.writer(f); w.writerow(['x_m','y_m','los_m','sigma_m'])
    for x,y,l in zip(XI,YI,LOS_OBS): w.writerow([f'{x:.1f}',f'{y:.1f}',f'{l:.6f}',S_I])
with open('gnss_obs.csv','w',newline='') as f:
    w=csv.writer(f); w.writerow(['station','x_m','y_m','uE_m','uN_m','uU_m','sig_h_m','sig_v_m'])
    for i in range(N_GNSS):
        w.writerow([f'G{i+1}',f'{XG[i]:.1f}',f'{YG[i]:.1f}',
                    f'{GE_O[i]:.6f}',f'{GN_O[i]:.6f}',f'{GU_O[i]:.6f}',S_H,S_V])

# ---------------------------------------------------------------- engines
def unpack(theta, use_ramp): return theta[:4], (theta[4:7] if use_ramp else np.zeros(3))
def residual(theta,use_insar,use_gnss,use_ramp,data):
    lo,ge_o,gn_o,gu_o = data; src,ramp = unpack(theta,use_ramp); r=[]
    if use_insar: r.append((insar_los(src,ramp,XI,YI)-lo)/S_I)
    if use_gnss:
        ge,gn,gu = mogi_enu(src,XG,YG); r += [(ge-ge_o)/S_H,(gn-gn_o)/S_H,(gu-gu_o)/S_V]
    return np.concatenate(r)
def n_params(use_insar,use_gnss,use_ramp): return 4+(3 if (use_insar and use_ramp) else 0)
def fit_ls(use_insar,use_gnss,data,use_ramp=True):
    use_ramp = use_ramp and use_insar; np_ = n_params(use_insar,use_gnss,use_ramp)
    lb,ub = (LB[:np_],UB[:np_]) if use_ramp else (LB[:4],UB[:4])
    x0 = np.array(CFG["ls_initial"])[:np_] if use_ramp else np.array(CFG["ls_initial"])[:4]
    sol = least_squares(residual,x0,bounds=(lb,ub),method="trf",x_scale="jac",ftol=1e-12,xtol=1e-12,
                        kwargs=dict(use_insar=use_insar,use_gnss=use_gnss,use_ramp=use_ramp,data=data))
    n_obs=sol.fun.size; dof=n_obs-np_; chi2r=float(np.sum(sol.fun**2))/dof
    cov=np.linalg.inv(sol.jac.T@sol.jac)*chi2r
    return dict(x=sol.x,sd=np.sqrt(np.diag(cov)),chi2r=chi2r,dof=dof,n_obs=n_obs,np=np_,status=int(sol.status))
def log_prob(theta,use_insar,use_gnss,use_ramp,data):
    lo,hi = LB[:theta.size],UB[:theta.size]
    if np.any(theta<lo) or np.any(theta>hi): return -np.inf
    return -0.5*np.sum(residual(theta,use_insar,use_gnss,use_ramp,data)**2)

def run_mcmc(use_insar,use_gnss,data,use_ramp=True,n_ens=1,seed0=1,dispersed=False,diag=True):
    use_ramp = use_ramp and use_insar; ndim = n_params(use_insar,use_gnss,use_ramp)
    ls = fit_ls(use_insar,use_gnss,data,use_ramp)
    nw,ns,bn,th = (CFG["mcmc"][k] for k in ("nwalkers","nsteps","burn","thin"))
    scale = np.array(CFG["mcmc_init_scales"])[:ndim]
    disp_rng = np.random.default_rng(CFG["seed"]+9999+seed0)
    ens_flat, ens_unthinned, accs, taus, centers = [],[],[],[],[]
    for e in range(n_ens):
        r = np.random.default_rng(CFG["seed"]+seed0+e); np.random.seed(CFG["seed"]+seed0+e)
        if dispersed and n_ens>1:
            center = np.clip(ls["x"] + disp_rng.normal(0,1,ndim)*CFG["mcmc"]["dispersion"]*scale,
                             LB[:ndim]+1e-6, UB[:ndim]-1e-6)
        else:
            center = ls["x"]
        centers.append(center)
        p0 = np.clip(center + r.normal(0,1,(nw,ndim))*scale, LB[:ndim]+1e-9, UB[:ndim]-1e-9)
        s = emcee.EnsembleSampler(nw,ndim,log_prob,
              kwargs=dict(use_insar=use_insar,use_gnss=use_gnss,use_ramp=use_ramp,data=data))
        s.run_mcmc(p0,ns,progress=False)
        ens_flat.append(s.get_chain(discard=bn,thin=th,flat=True))
        ens_unthinned.append(s.get_chain(discard=bn))
        accs.append(float(np.mean(s.acceptance_fraction)))
        try: taus.append(float(np.mean(s.get_autocorr_time(discard=bn,tol=0))))
        except Exception: taus.append(float('nan'))
    chain = np.vstack(ens_flat)
    L = min(a.shape[0] for a in ens_unthinned)
    A = np.concatenate([a[:L].transpose(1,0,2) for a in ens_unthinned], axis=0)  # (chains,draws,ndim)
    if diag:
        rhat = np.array([float(az.rhat(A[:,:,j])) for j in range(ndim)])
        ess  = np.array([float(az.ess (A[:,:,j])) for j in range(ndim)])
    else:
        rhat = np.full(ndim, np.nan); ess = np.full(ndim, np.nan)
    return dict(chain=chain, ls=ls, acc=float(np.mean(accs)), tau=float(np.nanmean(taus)),
                rhat=rhat, ess=ess, nsamp=chain.shape[0], n_chains=A.shape[0],
                draws=A.shape[1], centers=np.array(centers))

def summary(chain, cols=range(4)):
    q=np.percentile(chain[:,list(cols)],[16,50,84],axis=0)
    return q[1],q[1]-q[0],q[2]-q[1],q[2]-q[0]

def morans_I(vals, x, y, k=8):
    P=np.c_[x,y]; _,idx=cKDTree(P).query(P,k=k+1)
    z=vals-vals.mean(); n=len(vals); num=0.0
    for i in range(n):
        nb=idx[i,1:]; num += (1.0/len(nb))*np.sum(z[i]*z[nb])
    I=(n/n)*(num/np.sum(z*z))
    return float(I), float(-1.0/(n-1))

# ----- main joint (overdispersed) + ablation
DATA=(LOS_OBS,GE_O,GN_O,GU_O)
joint = run_mcmc(True,True,DATA,n_ens=CFG["mcmc"]["n_ensembles"],seed0=10,dispersed=True)
ins   = run_mcmc(True,False,DATA,n_ens=1,seed0=100,diag=False)
gns   = run_mcmc(False,True,DATA,n_ens=1,seed0=200,diag=False)
med,lo,hi,w68 = summary(joint["chain"])
rho = float(np.corrcoef(joint["chain"][:,2],joint["chain"][:,3])[0,1])

print(f"N_insar={N_INSAR} N_gnss={N_GNSS} LOS={CFG['los_enu_normalized']}")
print("LS joint:", {k:joint['ls'][k] for k in ('chi2r','dof','n_obs','np','status')})
print("MCMC joint: acc=%.2f tau=%.1f Rhat_max(src)=%.4f bulkESS_min(src)=%.0f n_chains=%d draws=%d nsamp=%d"
      % (joint["acc"],joint["tau"],joint["rhat"][:4].max(),joint["ess"][:4].min(),
         joint["n_chains"],joint["draws"],joint["nsamp"]))
print("dispersed ensemble centers (source):\n", np.round(joint["centers"][:,:4],0))
for i,nm in enumerate(PNAMES[:4]):
    print(f"  {nm:9s} true={MT[i]:>10.3g} med={med[i]:>10.3g} 68w={w68[i]:.3g} Rhat={joint['rhat'][i]:.4f} ESS={joint['ess'][i]:.0f}")
print("depth-dV rho =", round(rho,3))

def ablation_row(res,label):
    m,l,h,w = summary(res["chain"]); rr=float(np.corrcoef(res["chain"][:,2],res["chain"][:,3])[0,1])
    return dict(label=label,med=m.tolist(),w68=w.tolist(),chi2r=round(res["ls"]["chi2r"],3),rho_dV=round(rr,3))
ABL=[ablation_row(ins,"InSAR-only"),ablation_row(gns,"GNSS-only"),ablation_row(joint,"Joint")]
red_vs_insar=100*(1-ABL[2]["w68"][2]/ABL[0]["w68"][2]); red_vs_gnss=100*(1-ABL[2]["w68"][2]/ABL[1]["w68"][2])
print("depth 68% width:",{r["label"]:round(r["w68"][2]) for r in ABL},
      f"| red vs InSAR {red_vs_insar:.0f}% vs GNSS {red_vs_gnss:.0f}%")

# ----- MC calibration of LINEARIZED LS intervals
z68,z95 = norm.ppf(0.84),norm.ppf(0.975)
Nmc=CFG["mc"]["n_realizations"]; cov68=np.zeros(4); cov95=np.zeros(4)
biases=np.zeros((Nmc,4)); dwid=np.zeros((Nmc,3)); mrng=np.random.default_rng(CFG["seed"]+7)
for k in range(Nmc):
    dk=make_data(mrng); fj=fit_ls(True,True,dk); fi=fit_ls(True,False,dk); fg=fit_ls(False,True,dk)
    est,sd=fj["x"][:4],fj["sd"][:4]; biases[k]=est-MT
    cov68+=(np.abs(est-MT)<=z68*sd); cov95+=(np.abs(est-MT)<=z95*sd); dwid[k]=[fi["sd"][2],fg["sd"][2],fj["sd"][2]]
cov68/=Nmc; cov95/=Nmc; wid_red=100*(1-dwid[:,2]/dwid[:,0])
print("MC lin-LS coverage 68:",np.round(cov68,3)," 95:",np.round(cov95,3))
print("depth-width reduction vs InSAR: median %.0f%% [%.0f,%.0f]%%"
      %(np.median(wid_red),np.percentile(wid_red,16),np.percentile(wid_red,84)))

# ----- model mismatch (common noise draw for matched & two-source panels)
src2=np.array([-1200.0,1500.0,2600.0,1.2e6])
mm=np.random.default_rng(CFG["seed"]+3); LOSmm,GEmm,GNmm,GUmm=make_data(mm,src=MT,second=src2)
fit_mm=fit_ls(True,True,(LOSmm,GEmm,GNmm,GUmm))
mc=np.random.default_rng(CFG["seed"]+3); LOSma,GEma,GNma,GUma=make_data(mc,src=MT)
fit_ma=fit_ls(True,True,(LOSma,GEma,GNma,GUma))
def los_resid(fit,data): src,ramp=unpack(fit["x"],True); return data[0]-insar_los(src,ramp,XI,YI)
res_ma=los_resid(fit_ma,(LOSma,)); res_mm=los_resid(fit_mm,(LOSmm,))
def nn_corr(res): P=np.c_[XI,YI]; _,idx=cKDTree(P).query(P,k=2); return float(np.corrcoef(res,res[idx[:,1]])[0,1])
I_ma,EI=morans_I(res_ma,XI,YI); I_mm,_=morans_I(res_mm,XI,YI)
print("mismatch: chi2r matched=%.2f two-source=%.2f | Moran I matched=%.3f mismatch=%.3f (E=%.3f)"
      %(fit_ma["chi2r"],fit_mm["chi2r"],I_ma,I_mm,EI))
print("mismatch source bias:",np.round(fit_mm["x"][:4]-MT,1))

# reference-realization joint residual stats
srcj,rampj=unpack(joint["ls"]["x"],True); predj=insar_los(srcj,rampj,XI,YI); resj=(LOS_OBS-predj)*1000
RMSE=float(np.sqrt(np.mean(resj**2))); RMEAN=float(resj.mean()); RSTD=float(resj.std())
Ij,_=morans_I(resj/1000,XI,YI); NNj=nn_corr(resj/1000)
print(f"reference residual: mean={RMEAN:.2f} RMSE={RMSE:.2f} sd={RSTD:.2f} Moran_I={Ij:.3f} (E={EI:.3f}) NNcorr={NNj:.2f}")

# ===================================== FIGURES
plt.rcParams.update({'font.size':12,'axes.titlesize':12,'axes.labelsize':12,
                     'legend.fontsize':10,'lines.linewidth':2,'figure.dpi':150})
fig,ax=plt.subplots(figsize=(5.4,4.6))
sc=ax.scatter(XI/1000,YI/1000,c=LOS_OBS*1000,cmap='RdBu_r',s=22,vmin=-60,vmax=60)
q=ax.quiver(XG/1000,YG/1000,GE_O*1000,GN_O*1000,color='k',scale=90,width=0.006,zorder=5)
ax.quiverkey(q,0.80,1.06,20,'20 mm GNSS (horiz.)',labelpos='E',fontproperties={'size':9})
ax.scatter([MT[0]/1000],[MT[1]/1000],marker='*',s=240,c='#00d000',edgecolor='k',zorder=6,label='true source (x0,y0)')
for i in range(N_GNSS): ax.annotate(f'G{i+1}',(XG[i]/1000,YG[i]/1000),fontsize=7,ha='left',va='bottom')
cb=fig.colorbar(sc,ax=ax,shrink=0.9); cb.set_label('InSAR LOS incl. ramp (mm)')
ax.set_xlabel('East (km)'); ax.set_ylabel('North (km)'); ax.set_aspect('equal')
ax.set_title(f'L1 input:  N$_{{InSAR}}$={N_INSAR},  N$_{{GNSS}}$={N_GNSS}\nLOS $\\hat\\ell$=({LOS[0]:.3f}, {LOS[1]:.3f}, {LOS[2]:.3f})',fontsize=10)
ax.legend(loc='lower left',fontsize=8); fig.tight_layout(); fig.savefig('f6_data.png',dpi=200); plt.close(fig)

fig,axs=plt.subplots(1,3,figsize=(12,3.8))
for a,(dat,ttl,vm) in zip(axs,[(LOS_OBS*1000,'observed',60),(predj*1000,'modelled',60),(resj,'residual',20)]):
    s=a.scatter(XI/1000,YI/1000,c=dat,cmap='RdBu_r',s=24,vmin=-vm,vmax=vm)
    fig.colorbar(s,ax=a,shrink=0.85); a.set_title(f'{ttl} (mm)'); a.set_xlabel('East (km)'); a.set_aspect('equal')
axs[0].set_ylabel('North (km)')
axs[2].set_title(f"residual (mm)\nmean={RMEAN:.1f}, RMSE={RMSE:.1f}, sd={RSTD:.1f}, Moran's I={Ij:.2f} (E={EI:.2f})",fontsize=9)
fig.tight_layout(); fig.savefig('f7_fit.png',dpi=200); plt.close(fig)

lbls=[r'$x_0$ (m)',r'$y_0$ (m)',r'$d$ (m)',r'$\Delta V$ (m$^3$)']
figC=corner.corner(joint["chain"][:,:4],labels=lbls,truths=MT,quantiles=[0.16,0.5,0.84],
     show_titles=True,title_kwargs={'fontsize':11},label_kwargs={'fontsize':12},hist_kwargs={'linewidth':1.6})
figC.suptitle(f'Joint posterior (source params; ramp marginalized) — {joint["nsamp"]:,} samples',fontsize=12,y=1.02)
figC.savefig('f8_posterior.png',dpi=300,bbox_inches='tight'); plt.close(figC)

fig,axs=plt.subplots(1,2,figsize=(11,4.0))
axs[0].scatter(joint["chain"][:,2]/1000,joint["chain"][:,3]/1e6,s=2,alpha=0.12,color='0.25')
axs[0].axvline(MT[2]/1000,color='k',ls='--',lw=1.4); axs[0].axhline(MT[3]/1e6,color='k',ls='--',lw=1.4)
axs[0].set_xlabel('depth $d$ (km)'); axs[0].set_ylabel(r'$\Delta V$ (10$^6$ m$^3$)')
axs[0].set_title(f'depth–$\\Delta V$ ridge (Pearson r={rho:+.2f})')
for lab,lsz,c in [('InSAR-only','-','C3'),('GNSS-only','--','C2'),('Joint',':','C0')]:
    ch={'InSAR-only':ins,'GNSS-only':gns,'Joint':joint}[lab]["chain"]
    w={'InSAR-only':ABL[0],'GNSS-only':ABL[1],'Joint':ABL[2]}[lab]["w68"][2]
    axs[1].hist(ch[:,2]/1000,bins=70,density=True,histtype='step',ls=lsz,color=c,lw=2.2,label=f'{lab} (68% width {w:.0f} m)')
axs[1].axvline(MT[2]/1000,color='k',lw=1.2,label='true depth')
axs[1].set_xlabel('depth $d$ (km)'); axs[1].set_ylabel('posterior density')
axs[1].set_title('Joint InSAR–GNSS fusion narrows depth uncertainty'); axs[1].legend(fontsize=8)
fig.tight_layout(); fig.savefig('f9_tradeoff.png',dpi=200); plt.close(fig)

fig,axs=plt.subplots(1,2,figsize=(11,3.8)); xp=np.arange(4)
axs[0].bar(xp-0.2,cov68,0.4,label='68% interval',color='0.4'); axs[0].bar(xp+0.2,cov95,0.4,label='95% interval',color='0.75')
axs[0].axhline(0.68,ls='--',color='C3',lw=1.4); axs[0].axhline(0.95,ls=':',color='C0',lw=1.4)
axs[0].set_xticks(xp); axs[0].set_xticklabels(['$x_0$','$y_0$','$d$','$\\Delta V$'])
axs[0].set_ylabel('empirical coverage'); axs[0].set_ylim(0,1.05)
axs[0].set_title(f'Linearized LS interval coverage ({Nmc} realizations)'); axs[0].legend(fontsize=9)
axs[1].hist(wid_red,bins=30,color='0.5',edgecolor='k',lw=0.4)
axs[1].axvline(np.median(wid_red),color='C3',lw=2,ls='--',label=f'median {np.median(wid_red):.0f}%')
axs[1].set_xlabel('depth 68%-width reduction, joint vs InSAR-only (%)'); axs[1].set_ylabel('count')
axs[1].set_title('Fusion benefit across noise realizations'); axs[1].legend(fontsize=9)
fig.tight_layout(); fig.savefig('f10_calibration.png',dpi=200); plt.close(fig)

fig,axs=plt.subplots(1,2,figsize=(9.4,4.0))
for a,(r,ttl,chi,I) in zip(axs,[(res_ma*1000,'matched model',fit_ma['chi2r'],I_ma),
                                 (res_mm*1000,'two-source truth (mismatch)',fit_mm['chi2r'],I_mm)]):
    s=a.scatter(XI/1000,YI/1000,c=r,cmap='RdBu_r',s=24,vmin=-40,vmax=40); fig.colorbar(s,ax=a,shrink=0.85)
    a.set_title(f"{ttl}\nreduced $\\chi^2$={chi:.2f}, Moran's I={I:.2f}",fontsize=10)
    a.set_xlabel('East (km)'); a.set_aspect('equal')
axs[0].set_ylabel('North (km)'); fig.tight_layout(); fig.savefig('f11_mismatch.png',dpi=200); plt.close(fig)

# ===================================== RECORDS
def pm(i): return f"{med[i]:.0f} (-{lo[i]:.0f}/+{hi[i]:.0f})"
record = {
  "schema":{"version":"2.1","fields":["robust_parameters","trade_off_dominated_parameters",
    "dominant_data_driver","plausible_alternatives","limits_of_interpretation",
    "fit_quality","convergence","calibration","model_mismatch","provenance"]},
  "robust_parameters":{"x0_m":pm(0),"y0_m":pm(1),
     "note":"horizontal position ~+/-85 m (equal-tailed central 68% CI half-width)"},
  "trade_off_dominated_parameters":{"d_m":pm(2),"dV_m3":pm(3),
     "correlation_depth_dV":round(rho,3),"note":"resolvable only jointly along a correlated ridge"},
  "dominant_data_driver":{
     "depth":f"GNSS 3-D reduces depth 68% width from {ABL[0]['w68'][2]:.0f} m (InSAR-only) to {ABL[2]['w68'][2]:.0f} m (joint)",
     "horizontal":"InSAR LOS pattern constrains x0,y0 and the deformation shape",
     "evidence":"per-parameter sensor-removal ablation (Table 9)"},
  "plausible_alternatives":"shallower-smaller vs deeper-larger sources along the depth-dV ridge",
  "limits_of_interpretation":"homogeneous isotropic elastic half-space, single point source, static snapshot, "
     "fixed Poisson ratio; same-model synthetic verification only; NOT a hazard product",
  "fit_quality":{"reduced_chi2":round(joint["ls"]["chi2r"],3),"dof":joint["ls"]["dof"],
     "n_obs":joint["ls"]["n_obs"],"n_params_incl_nuisance":joint["ls"]["np"],
     "residual_RMSE_mm":round(RMSE,1),"residual_mean_mm":round(RMEAN,2),"residual_sd_mm":round(RSTD,1),
     "residual_Moran_I":round(Ij,3),"residual_Moran_I_null":round(EI,3),
     "residual_NN_corr_exploratory":round(NNj,3)},
  "convergence":{
     "diagnostic":"rank-normalized split-R-hat and bulk-ESS (ArviZ) over walker-chains from 4 overdispersed, independently seeded ensembles",
     "rank_normalized_split_Rhat_max_source":round(float(joint["rhat"][:4].max()),4),
     "bulk_ESS_min_source":round(float(joint["ess"][:4].min())),
     "mean_acceptance_fraction":round(joint["acc"],3),
     "integrated_autocorr_time_steps":round(joint["tau"],1),
     "n_walker_chains":int(joint["n_chains"]),"draws_per_chain":int(joint["draws"]),
     "n_ensembles":CFG["mcmc"]["n_ensembles"],"overdispersed_starts":True,
     "n_posterior_samples_thinned":int(joint["nsamp"])},
  "calibration":{"n_realizations":Nmc,
     "interval_type":"linearized least-squares normal intervals (NOT posterior credible intervals)",
     "coverage_68":{n:round(float(c),3) for n,c in zip(['x0','y0','d','dV'],cov68)},
     "coverage_95":{n:round(float(c),3) for n,c in zip(['x0','y0','d','dV'],cov95)},
     "depth_width_reduction_pct_median":round(float(np.median(wid_red)),1)},
  "model_mismatch":{"chi2r_matched":round(fit_ma["chi2r"],3),"chi2r_two_source":round(fit_mm["chi2r"],3),
     "Moran_I_matched":round(I_ma,3),"Moran_I_mismatch":round(I_mm,3),
     "source_bias_x0y0d_dV":[round(float(v),1) for v in (fit_mm["x"][:4]-MT)],
     "note":"structural error evidenced jointly by residual organization, Moran's I, and parameter bias, "
            "not by the modest chi2 change alone"},
  "provenance":"generated by or3_demo.py v3.0; seed "+str(CFG["seed"])}
json.dump(record,open('decision_record.json','w'),indent=2)

diagnostics = {
  "reference_residual":{"mean_mm":round(RMEAN,3),"RMSE_mm":round(RMSE,3),"sd_mm":round(RSTD,3),
     "Moran_I":round(Ij,4),"Moran_I_null":round(EI,4),"NN_corr":round(NNj,4)},
  "mcmc":{"Rhat_source":[round(float(v),4) for v in joint["rhat"][:4]],
          "bulk_ESS_source":[round(float(v)) for v in joint["ess"][:4]],
          "acceptance":round(joint["acc"],3),"autocorr_time":round(joint["tau"],1),
          "ensemble_centers_source":np.round(joint["centers"][:,:4],1).tolist()},
  "mismatch":{"chi2r_matched":round(fit_ma["chi2r"],3),"chi2r_mismatch":round(fit_mm["chi2r"],3),
              "Moran_I_matched":round(I_ma,4),"Moran_I_mismatch":round(I_mm,4)}}
json.dump(diagnostics,open('diagnostics.json','w'),indent=2)
json.dump(CFG,open('config.json','w'),indent=2,default=float)
json.dump({"ablation":ABL,"reduction_vs_InSAR_pct":round(red_vs_insar,1),
           "reduction_vs_GNSS_pct":round(red_vs_gnss,1)},open('ablation.json','w'),indent=2)

with open('requirements_lock.txt','w') as f:
    f.write(f"# platform: {platform.platform()}\n# python: {platform.python_version()}\n")
    f.write(f"numpy=={np.__version__}\nscipy=={scipy.__version__}\nmatplotlib=={matplotlib.__version__}\n"
            f"emcee=={emcee.__version__}\ncorner=={corner.__version__}\narviz=={az.__version__}\n")
print("\nWritten: f6-f11, insar/gnss_obs.csv, config.json, decision_record.json, ablation.json, diagnostics.json, requirements_lock.txt")
