import pathlib,subprocess,os,json,shutil,time
p=pathlib.Path('/workspace/diagnostic');repo=p/'repo';env=os.environ.copy();env.update(PYTHONPATH=str(repo/'src'),MAX_JOBS='4',TORCH_CUDA_ARCH_LIST='9.0',PIP_DISABLE_PIP_VERSION_CHECK='1');status=[]
def run(name,cmd,limit=1200):
 with (p/(name+'.log')).open('w') as f:
  try:r=subprocess.run(cmd,cwd=repo if repo.exists() else p,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=limit);code=r.returncode
  except subprocess.TimeoutExpired:code=124
 status.append(dict(name=name,exit_code=code));(p/'accept-status.json').write_text(json.dumps(status,indent=2));print(name,code,flush=True);return code
run('clone',['git','clone','-b','portfolio/paged-gqa-decode','https://github.com/ensigncode/paged-decode-lab.git',str(repo)])
run('deps',['python','-m','pip','install','pytest','ninja','flashinfer-python==0.7.2.post1'])
run('full-pytest',['python','-m','pytest','-q','-rA','--junitxml='+str(p/'pytest.xml')])
tool=(p/'new-tool-path.txt').read_text()
run('old-custom-memcheck',['compute-sanitizer','--tool','memcheck','--error-exitcode','1','python','-m','pytest','-q','tests/test_cuda.py'])
for t in ['memcheck','racecheck','synccheck','initcheck']:
 run('new-'+t,[tool,'--tool',t,'--error-exitcode','1','python','-m','pytest','-q','tests/test_cuda.py'])
import hashlib
names=subprocess.check_output(['git','ls-files'],cwd=repo,text=True).splitlines();(p/'source-hashes.json').write_text(json.dumps({f:hashlib.sha256((repo/f).read_bytes()).hexdigest() for f in names},indent=2))
run('freeze',['python','-m','pip','freeze']);run('revision',['git','rev-parse','HEAD']);(p/'ACCEPT_DONE').write_text(time.ctime());print('ACCEPT_DONE',flush=True)
