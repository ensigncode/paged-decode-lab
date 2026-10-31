import json,pathlib,urllib.request,tarfile,hashlib,subprocess,os
p=pathlib.Path('/workspace/diagnostic');u='https://developer.download.nvidia.com/compute/cuda/redist/';data=urllib.request.urlopen(u+'redistrib_13.4.3.json',timeout=30).read();(p/'latest-redistrib.json').write_bytes(data);e=json.loads(data)['cuda_sanitizer_api']['linux-x86_64'];a=p/'latest.tar.xz';urllib.request.urlretrieve(u+e['relative_path'],a);assert hashlib.sha256(a.read_bytes()).hexdigest()==e['sha256']
with tarfile.open(a) as t:t.extractall(p/'latest',filter='data')
tool=next(f for f in (p/'latest').rglob('compute-sanitizer') if f.is_file());tool.chmod(0o755);(p/'latest-tool-path.txt').write_text(str(tool));env=os.environ.copy();env.update(PYTHONPATH=str(p/'repo/src'),MAX_JOBS='4',TORCH_CUDA_ARCH_LIST='9.0');status=[]
for name,cmd in [('latest-version',[str(tool),'--version']),('latest-negative',[str(tool),'--tool','memcheck','--error-exitcode','1',str(p/'negative-control')])]+[('latest-'+t,[str(tool),'--tool',t,'--error-exitcode','1','python','-m','pytest','-q','tests/test_cuda.py']) for t in ['memcheck','racecheck','synccheck','initcheck']]:
 with (p/(name+'.log')).open('w') as f:r=subprocess.run(cmd,cwd=p/'repo',env=env,stdout=f,stderr=subprocess.STDOUT,timeout=600)
 status.append(dict(name=name,exit_code=r.returncode));(p/'latest-status.json').write_text(json.dumps(status,indent=2));print(name,r.returncode,flush=True)
print('LATEST_DONE',flush=True)
