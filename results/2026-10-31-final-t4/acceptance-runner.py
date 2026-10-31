# Final published revision acceptance. Historical cells and outputs are retained.
import os, sys, subprocess, pathlib, json, hashlib, zipfile, time
p=pathlib.Path('/content/ensigncode-final-t4');p.mkdir(exist_ok=False)
status=[]
def run(name,cmd,limit=1200):
 print('START',name,flush=True)
 with (p/(name+'.log')).open('w') as f:
  try: c=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,timeout=limit,env=env,cwd=repo if repo.exists() else p).returncode
  except subprocess.TimeoutExpired:c=124
 status.append({'name':name,'exit_code':c});(p/'status.json').write_text(json.dumps(status,indent=2));print('DONE',name,c,flush=True);return c
repo=p/'repo';env=os.environ.copy();env.update(MAX_JOBS='2',TORCH_CUDA_ARCH_LIST='7.5',PYTHONPATH=str(repo/'src'))
assert run('clone',['git','clone','--branch','portfolio/paged-gqa-decode','https://github.com/ensigncode/paged-decode-lab.git',str(repo)])==0
assert run('checkout',['git','checkout','41a672407c7185c8d5c86bb9dc50ef69031c3478'])==0
assert run('dependencies',[sys.executable,'-m','pip','install','ninja','pytest','flashinfer-python==0.7.2.post1'])==0
run('nvidia-smi',['nvidia-smi']);run('nvcc',['nvcc','--version']);run('sanitizer-version',['compute-sanitizer','--version'])
run('environment',[sys.executable,'-c','import torch,sys;print(sys.version);print(torch.__version__,torch.version.cuda,torch.cuda.get_device_name(0))'])
h={str(f.relative_to(repo)):hashlib.sha256(f.read_bytes()).hexdigest() for folder in ['src','tests','benchmarks'] for f in (repo/folder).rglob('*') if f.is_file() and '__pycache__' not in str(f)};(p/'source-sha256.json').write_text(json.dumps(h,indent=2))
run('pytest',[sys.executable,'-m','pytest','-q','--junitxml='+str(p/'pytest.xml')])
for tool in ['memcheck','racecheck','synccheck','initcheck']:
 run(tool,['compute-sanitizer','--tool',tool,'--error-exitcode','1',sys.executable,'-m','pytest','-q','tests/test_cuda.py'])
s=(repo/'results/2026-10-31-h100/diagnostic/stress-clean.py').read_text().replace('/workspace/diagnostic/stress-clean-progress.json',str(p/'stress-progress.json'))
(p/'stress.py').write_text(s);run('random-stress',[sys.executable,str(p/'stress.py')])
for b in [1,4]:
 run('competitive-b'+str(b),[sys.executable,'benchmarks/bench_competitive.py','--batch',str(b),'--context','4096','--output',str(p/f'competitive-b{b}.json')])
run('freeze',[sys.executable,'-m','pip','freeze'])
(p/'FINISHED.json').write_text(json.dumps({'all_commands_passed':all(x['exit_code']==0 for x in status),'revision':'41a672407c7185c8d5c86bb9dc50ef69031c3478','status':status},indent=2))
artifacts=[f for f in p.iterdir() if f.is_file()];(p/'artifact-sha256.json').write_text(json.dumps({f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in artifacts},indent=2))
z=p.parent/'Ensigncode_Final_T4_Evidence.zip'
with zipfile.ZipFile(z,'w',zipfile.ZIP_DEFLATED) as a:
 for f in p.iterdir():
  if f.is_file():a.write(f,f.name)
print('FINAL_T4_COMPLETE',flush=True)
from google.colab import files
files.download(str(z))