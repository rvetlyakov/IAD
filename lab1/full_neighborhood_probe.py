import csv
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
from hybrid_ga import Config, best_columns, costs, improve_swaps, run
from scipy.optimize import linear_sum_assignment
from three_cycle import improve_three_cycle

rows=[]
base=Config(population=64,generations=100,selection='rank',local_every=5,local_top=2,local_candidates=16,cycle3=True,cycle3_passes=3,inversion_pct=1)
for n in (50,150,300,450):
 for rep in range(3):
  seed=930000000+n+10003*rep;a=np.random.default_rng(seed).integers(1,101,(n,n),dtype=np.int32)
  r,c=linear_sum_assignment(a);opt=int(a[r,c].sum());ans=run(a,base,seed+17)
  for k in (16,32,64,n):
   p=ans['best'].copy();cols=best_columns(a,k)
   p,swaps=improve_swaps(a,p,cols,20);p,cycles=improve_three_cycle(a,p,cols,20)
   rows.append({'n': n,'rep': rep,'k': k,'before': ans['cost'],'after': int(costs(a,p)[0]),'exact': opt,'swaps': swaps,'cycles': cycles})
 print(n,flush=True)
with open('results/class_diagnosis_20261028/full_neighborhood_probe.csv','w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
