import os,pathlib,subprocess,json,hashlib
p=pathlib.Path('/workspace/diagnostic');prefix=p/'cuda130';venv=p/'cu130env';py=str(venv/'bin/python');env=os.environ.copy();env.update(CUDA_HOME=str(prefix),PATH=str(venv/'bin')+':'+str(prefix/'bin')+':'+env['PATH'],MAX_JOBS='4',TORCH_CUDA_ARCH_LIST='9.0',PYTHONPATH=str(p/'repo/src'),TORCH_EXTENSIONS_DIR=str(p/'clean-extensions'));status=[]
tool=(p/'latest-tool-path.txt').read_text()
commands=[('portable-pytest',[py,'-m','pytest','-q','-rA','--maxfail=1','--junitxml='+str(p/'portable-pytest.xml')])]+[('portable-'+t,[tool,'--tool',t,'--error-exitcode','1',py,'-m','pytest','-q','tests/test_cuda.py','--maxfail=1']) for t in ['memcheck','racecheck','synccheck','initcheck']]
for name,cmd in commands:
 with (p/(name+'.log')).open('w') as f:r=subprocess.run(cmd,cwd=p/'repo',env=env,stdout=f,stderr=subprocess.STDOUT,timeout=1200)
 status.append(dict(name=name,exit_code=r.returncode));(p/'portable-status.json').write_text(json.dumps(status,indent=2));print(name,r.returncode,flush=True)
 if name=='portable-pytest' and r.returncode:break
(p/'portable-source-diff.patch').write_text(subprocess.check_output(['git','diff'],cwd=p/'repo',text=True));(p/'portable-source-sha256.json').write_text(json.dumps({f:hashlib.sha256((p/'repo'/f).read_bytes()).hexdigest() for f in subprocess.check_output(['git','ls-files','src','tests','benchmarks'],cwd=p/'repo',text=True).splitlines()},indent=2));print('PORTABLE_RETEST_DONE',flush=True)
