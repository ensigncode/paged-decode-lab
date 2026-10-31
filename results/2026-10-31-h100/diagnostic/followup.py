from pathlib import Path
exec(Path('/workspace/diagnostic/retest.py').read_text().split('commands=')[0])
assert all(s['exit_code']==0 for s in json.loads((p/'portable-status.json').read_text()))
commands=[('portable-stress',[py,str(p/'stress-clean.py')])]+[(f'portable-bench-b{b}',[py,'benchmarks/bench_competitive.py','--batch',str(b),'--context','4096','--output',str(p/f'portable-bench-b{b}.json')]) for b in [1,4,16]]
for name,cmd in commands:
 with (p/(name+'.log')).open('w') as f:r=subprocess.run(cmd,cwd=p/'repo',env=env,stdout=f,stderr=subprocess.STDOUT,timeout=1200)
 status.append(dict(name=name,exit_code=r.returncode));(p/'followup-status.json').write_text(json.dumps(status,indent=2));print(name,r.returncode,flush=True)
print('FOLLOWUP_DONE',flush=True)
