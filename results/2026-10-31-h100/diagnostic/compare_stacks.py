from pathlib import Path
exec(Path('/workspace/diagnostic/retest.py').read_text().split('commands=')[0])
original=p/'original-repo'
subprocess.run(['git','worktree','add','--detach',str(original),'HEAD'],cwd=p/'repo',check=True,stdout=subprocess.DEVNULL)
env['CPATH']=str(venv/'lib/python3.12/site-packages/nvidia/cu13/include');env['PYTHONPATH']=str(original/'src');env['TORCH_EXTENSIONS_DIR']=str(p/'original-clean-extensions')
with (p/'original-clean-memcheck.log').open('w') as f:r=subprocess.run([tool,'--tool','memcheck','--error-exitcode','1',py,'-m','pytest','-q','tests/test_cuda.py','--maxfail=1'],cwd=original,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=600)
(p/'original-clean-status.json').write_text(json.dumps(dict(exit_code=r.returncode)));print('ORIGINAL_CLEAN',r.returncode,flush=True)
legacy=os.environ.copy();legacy.update(PYTHONPATH=str(p/'repo/src'),MAX_JOBS='4',TORCH_CUDA_ARCH_LIST='9.0')
with (p/'portable-legacy-memcheck.log').open('w') as f:r=subprocess.run([tool,'--tool','memcheck','--error-exitcode','1','python','-m','pytest','-q','tests/test_cuda.py','--maxfail=1'],cwd=p/'repo',env=legacy,stdout=f,stderr=subprocess.STDOUT,timeout=600)
(p/'portable-legacy-status.json').write_text(json.dumps(dict(exit_code=r.returncode)));print('PORTABLE_LEGACY',r.returncode,flush=True);print('COMPARISON_DONE',flush=True)
