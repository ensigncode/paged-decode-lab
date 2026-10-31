import subprocess,pathlib,json,time,os,urllib.request,tarfile
p=pathlib.Path('/workspace/diagnostic');os.chdir(p);status=[]
def run(name,cmd,timeout=300):
 with (p/(name+'.log')).open('w') as f:
  try:r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,timeout=timeout);c=r.returncode
  except subprocess.TimeoutExpired:c=124
 status.append(dict(name=name,exit_code=c));(p/'status.json').write_text(json.dumps(status,indent=2));print(name,c,flush=True);return c
run('gpu',['nvidia-smi']);run('old-version',['compute-sanitizer','--version']);run('compile',['nvcc','-arch=sm_90','-lineinfo','probe.cu','-o','probe']);run('old-probe',['compute-sanitizer','--tool','memcheck','--error-exitcode','1','./probe'])
try:
 url='https://developer.download.nvidia.com/compute/cuda/redist/redistrib_13.2.2.json'
 data=urllib.request.urlopen(url,timeout=30).read();(p/'redistrib.json').write_bytes(data);index=json.loads(data);entry=index['cuda_sanitizer_api']['linux-x86_64'];rel=entry['relative_path'];archive=p/'sanitizer.tar.xz';urllib.request.urlretrieve('https://developer.download.nvidia.com/compute/cuda/redist/'+rel,archive)
 import hashlib
 assert hashlib.sha256(archive.read_bytes()).hexdigest()==entry['sha256']
 with tarfile.open(archive) as t:t.extractall(p/'new',filter='data')
 bins=list((p/'new').rglob('compute-sanitizer'));assert bins,bins
 tool=str(bins[0]);(p/'new-tool-path.txt').write_text(tool);run('new-version',[tool,'--version']);run('new-probe',[tool,'--tool','memcheck','--error-exitcode','1','./probe'])
except Exception as e:(p/'tool-install-error.txt').write_text(repr(e));print(repr(e),flush=True)
print('DIAGNOSTIC_READY',flush=True)
