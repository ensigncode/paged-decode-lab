import os,sys,subprocess,json,time,hashlib,pathlib,traceback,shutil
base=pathlib.Path('/workspace/ensigncode-h100');base.mkdir(exist_ok=True)
repo=base/'repo';evidence=base/'evidence';evidence.mkdir(exist_ok=True)
status=[]
env=os.environ.copy();env.update(MAX_JOBS='4',TORCH_CUDA_ARCH_LIST='9.0',PYTHONPATH=str(repo/'src'),PIP_DISABLE_PIP_VERSION_CHECK='1')
def run(name,cmd,timeout=1800,cwd=None):
    t=time.time()
    with (evidence/(name+'.log')).open('w') as log:
        try:
            result=subprocess.run(cmd,cwd=cwd or (repo if repo.exists() else base),env=env,stdout=log,stderr=subprocess.STDOUT,timeout=timeout)
            code=result.returncode
        except subprocess.TimeoutExpired: code=124
    status.append(dict(name=name,exit_code=code,seconds=time.time()-t))
    (evidence/'status.json').write_text(json.dumps(status,indent=2));print(name,code,flush=True)
    return code==0
try:
    if not repo.exists():
        assert run('clone',['git','clone','--branch','portfolio/paged-gqa-decode','https://github.com/ensigncode/paged-decode-lab.git',str(repo)],300)
    revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
    assert revision=='2b58e7d6f7ff1221a440cc08ace7211d2c5d4406',revision
    import torch
    assert 'H100' in torch.cuda.get_device_name(0)
    metadata=dict(revision=revision,python=sys.version,torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0),capability=torch.cuda.get_device_capability(0))
    (evidence/'environment.json').write_text(json.dumps(metadata,indent=2))
    run('nvidia-smi',['nvidia-smi']);run('nvcc',['nvcc','--version']);run('pip-freeze-before',[sys.executable,'-m','pip','freeze'])
    assert run('dependencies',[sys.executable,'-m','pip','install','pytest','ninja','flashinfer-python==0.7.2.post1'],1800)
    names=subprocess.check_output(['git','ls-files'],cwd=repo,text=True).splitlines()
    (evidence/'source-sha256.json').write_text(json.dumps({p:hashlib.sha256((repo/p).read_bytes()).hexdigest() for p in names},indent=2))
    ok=run('pytest',[sys.executable,'-m','pytest','-q','-rA','--junitxml='+str(evidence/'pytest.xml')],2400)
    for tool in ['memcheck','racecheck','synccheck','initcheck']:
        run(tool,['compute-sanitizer','--tool',tool,'--error-exitcode','1',sys.executable,'-m','pytest','-q','tests/test_cuda.py','--maxfail=1'],1800)
    if ok:
        run('random-stress',[sys.executable,str(base/'stress.py')],2400)
        for b in [1,4,16]:
            for l in [1024,4096,16384]:
                name=f'competitive-b{b}-l{l}'
                run(name,[sys.executable,'benchmarks/bench_competitive.py','--batch',str(b),'--context',str(l),'--output',str(evidence/(name+'.json'))],1800)
    run('pip-freeze-after',[sys.executable,'-m','pip','freeze'])
except Exception:
    (evidence/'runner-exception.log').write_text(traceback.format_exc());print(traceback.format_exc(),flush=True)
finally:
    (evidence/'FINISHED.json').write_text(json.dumps({'finished_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'all_commands_passed':bool(status) and all(s['exit_code']==0 for s in status)},indent=2))
    shutil.make_archive(str(base/'h100-evidence'),'zip',evidence)
    os.sync();print('EVIDENCE_READY',flush=True)
