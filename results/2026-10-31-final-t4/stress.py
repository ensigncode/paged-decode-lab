import numpy as np,torch,json,time
from paged_decode.runtime import DecodePlan
from paged_decode.oracle import dense_attention
from paged_decode.workload import make_metadata,make_tensors
from pathlib import Path
p=Path('/content/ensigncode-final-t4/stress-progress.json');rng=np.random.default_rng(97531);worst=0;start=time.time()
for seed in range(1000):
 b=int(rng.choice([1,2,3]));kv=int(rng.choice([1,2,4]));h=kv*int(rng.choice([1,2,4,8]));d=int(rng.choice([64,128]));page=int(rng.choice([16,32]));length=int(rng.choice([1,15,16,17,31,32,33,127,129,257,513]))
 layout,table,lengths=make_metadata(b,h,kv,d,length,page,seed=seed)
 if seed%3==0:lengths[-1]=0;table[-1,:]=-1
 if seed%5==0:table[table>=0]=0
 dtype=torch.float16 if seed%2 else torch.float32
 q,k,v=make_tensors(layout,dtype=dtype,seed=seed+10000)
 if seed%7==0:q.mul_(20);k.mul_(20)
 ref=dense_attention(*(t.cpu().numpy() for t in [q,k,v]),table,lengths)
 for merge in ['serial','shared']:
  plan=DecodePlan(layout,table,lengths,split_tokens=int(rng.choice([1,7,31,128,256,511])),merge=merge)
  out=plan.run(q,k,v).cpu().numpy();err=float(np.max(np.abs(out-ref)));worst=max(worst,err);np.testing.assert_allclose(out,ref,atol=0.003,rtol=0.003)
 p.write_text(json.dumps(dict(completed_cases=seed+1,merge_executions=2*(seed+1),max_abs=worst,seconds=time.time()-start)))
print(p.read_text())
