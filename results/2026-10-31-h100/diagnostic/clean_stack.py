import pathlib,subprocess,os,json,urllib.request,tarfile,hashlib,shutil
p=pathlib.Path('/workspace/diagnostic');prefix=p/'cuda130';venv=p/'cu130env';status=[]
def run(name,cmd,env=None,limit=1200):
 with (p/(name+'.log')).open('w') as f:
  try:r=subprocess.run(cmd,cwd=p/'repo',env=env,stdout=f,stderr=subprocess.STDOUT,timeout=limit);c=r.returncode
  except subprocess.TimeoutExpired:c=124
 status.append(dict(name=name,exit_code=c));(p/'clean-status.json').write_text(json.dumps(status,indent=2));print(name,c,flush=True);return c
assert run('venv',['python','-m','venv',str(venv)])==0
py=str(venv/'bin/python')
assert run('torch130-install',[py,'-m','pip','install','torch==2.11.0','--index-url','https://download.pytorch.org/whl/cu130'],limit=1800)==0
assert run('clean-deps',[py,'-m','pip','install','numpy','pytest','ninja','flashinfer-python==0.7.2.post1'],limit=1200)==0
u='https://developer.download.nvidia.com/compute/cuda/redist/';data=urllib.request.urlopen(u+'redistrib_13.0.2.json',timeout=30).read();(p/'cuda130-redistrib.json').write_bytes(data);idx=json.loads(data);prefix.mkdir(exist_ok=True)
for key in ['cuda_nvcc','cuda_crt','cuda_cudart','cuda_cccl','libnvvm']:
 e=idx[key]['linux-x86_64'];a=p/(key+'.tar.xz');urllib.request.urlretrieve(u+e['relative_path'],a);assert hashlib.sha256(a.read_bytes()).hexdigest()==e['sha256']
 dest=p/('extract-'+key)
 with tarfile.open(a) as t:t.extractall(dest,filter='data')
 folder=next(dest.iterdir());shutil.copytree(folder,prefix,dirs_exist_ok=True)
for folder in ['bin','nvvm/bin']:
 if (prefix/folder).exists():
  for f in (prefix/folder).iterdir():
   if f.is_file():f.chmod(f.stat().st_mode|0o111)
env=os.environ.copy();env.update(CUDA_HOME=str(prefix),PATH=str(venv/'bin')+':'+str(prefix/'bin')+':'+env['PATH'],MAX_JOBS='4',TORCH_CUDA_ARCH_LIST='9.0',PYTHONPATH=str(p/'repo/src'),TORCH_EXTENSIONS_DIR=str(p/'clean-extensions'))
run('clean-environment',[py,'-c','import torch,sys;print(sys.version);print(torch.__version__,torch.version.cuda,torch.cuda.get_device_name(0))'],env)
run('clean-nvcc',[str(prefix/'bin/nvcc'),'--version'],env)
run('clean-pytest',[py,'-m','pytest','-q','-rA','--junitxml='+str(p/'clean-pytest.xml')],env,1800)
tool=(p/'latest-tool-path.txt').read_text()
for t in ['memcheck','racecheck','synccheck','initcheck']:
 run('clean-'+t,[tool,'--tool',t,'--error-exitcode','1',py,'-m','pytest','-q','tests/test_cuda.py'],env,900)
run('clean-freeze',[py,'-m','pip','freeze'],env);print('CLEAN_STACK_DONE',flush=True)
