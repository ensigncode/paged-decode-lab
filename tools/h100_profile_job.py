import pathlib,subprocess,os,json,hashlib,time,shutil,zipfile
root=pathlib.Path("/workspace/h100-profile");root.mkdir()
repo=root/"repo";state=[];env=os.environ.copy();started=time.monotonic()
shutil.copy2(__file__,root/"runner.py")
def run(name,command,limit=600):
 print("START",name,flush=True)
 with (root/(name+".log")).open("w") as log:
  try:code=subprocess.run(command,cwd=repo if repo.exists() else root,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=max(1,min(limit,1200-time.monotonic()+started))).returncode
  except subprocess.TimeoutExpired:code=124
 state.append(dict(name=name,exit_code=code));(root/"status.json").write_text(json.dumps(state,indent=2));print("DONE",name,code,flush=True)
 return code
try:
 assert run("clone",["git","clone","-b","portfolio/paged-gqa-decode","https://github.com/ensigncode/paged-decode-lab.git",str(repo)])==0
 assert run("checkout",["git","checkout","1d7144b86bf577bf9dcb606deb7fef1a5a32819c"])==0
 py=str(root/"venv/bin/python")
 assert run("venv",["python","-m","venv",str(root/"venv")])==0
 assert run("torch-install",[py,"-m","pip","install","--no-cache-dir","torch==2.11.0","--index-url","https://download.pytorch.org/whl/cu130"])==0
 assert run("dependencies",[py,"-m","pip","install","--no-cache-dir","numpy","pytest","ninja","flashinfer-python==0.7.2.post1"])==0
 env.update(CUDA_HOME="/usr/local/cuda",MAX_JOBS="4",TORCH_CUDA_ARCH_LIST="9.0",PYTHONPATH=str(repo/"src"))
 env["PATH"]=str(root/"venv/bin")+":/usr/local/cuda/bin:"+env["PATH"]
 run("environment",[py,"-c","import torch,sys;print(sys.version);print(torch.__version__,torch.version.cuda,torch.cuda.get_device_name(0))"])
 run("gpu",["nvidia-smi"]);run("nvcc",["nvcc","--version"]);run("freeze",[py,"-m","pip","freeze"])
 assert run("pytest",[py,"-m","pytest","-q","--junitxml="+str(root/"pytest.xml")])==0
 assert run("profile",[py,"benchmarks/profile_decode.py","--output",str(root/"profiles")])==0
 tool=shutil.which("ncu")
 if tool:
  run("ncu-version",[tool,"--version"])
  if run("ncu",[tool,"--nvtx","--nvtx-include","profile/","--set","detailed","--launch-count","2","--export",str(root/"ncu-report"),py,"benchmarks/profile_decode.py","--ncu-target","--output",str(root/"ncu-target")],180)==0:
   for page in ["details","raw"]:run("ncu-"+page,[tool,"--import",str(root/"ncu-report.ncu-rep"),"--page",page,"--csv"],60)
 else:(root/"ncu-unavailable.txt").write_text("ncu not found in PATH; no hardware counters collected.")
except Exception as error:(root/"error.txt").write_text(repr(error));print("ERROR",repr(error),flush=True)
finally:
 names=subprocess.check_output(["git","ls-files","src","tests","benchmarks"],cwd=repo,text=True).splitlines()
 (root/"source-sha256.json").write_text(json.dumps({n:hashlib.sha256((repo/n).read_bytes()).hexdigest() for n in names},indent=2))
 files=[f for f in root.rglob("*") if f.is_file() and repo not in f.parents and root/"venv" not in f.parents]
 (root/"artifact-sha256.json").write_text(json.dumps({str(f.relative_to(root)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files},indent=2))
 files.append(root/"artifact-sha256.json")
 with zipfile.ZipFile(root/"evidence.zip","w",zipfile.ZIP_DEFLATED) as archive:
  for f in files:archive.write(f,str(f.relative_to(root)))
 print("EVIDENCE_READY",hashlib.sha256((root/"evidence.zip").read_bytes()).hexdigest(),flush=True)
