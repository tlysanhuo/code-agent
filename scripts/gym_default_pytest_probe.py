"""CPU-only DSH shell gate for repository-default pytest options; no model score."""
import json
from pathlib import Path
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from gym_prepare import BASE, make_case, task
from gym_facility import execute
from swe_prepare import write


def main():
    records=[json.loads(s) for s in (BASE/'validation-attempts.jsonl').read_text().splitlines()]
    out=BASE/'default-pytest-probes'/str(time.time_ns())
    results=[]
    for instance,filename in [('dask__dask-10380','dask/tests/test_cli.py'),('pydantic__pydantic-9082','tests/test_fields.py')]:
        record=next(r for r in reversed(records) if r['instance_id']==instance and r['passed'])
        case=out/instance;make_case(case,task(instance),Path(record['python']))
        calls=[]
        class Handler(BaseHTTPRequestHandler):
            def log_message(self,*args):pass
            def do_POST(self):
                body=json.loads(self.rfile.read(int(self.headers['Content-Length'])));calls.append(body)
                if len(calls)==1:
                    cmd=f'python -m pytest {filename} -q; result=$?; printf "\\nDEFAULT_PYTEST_EXIT=%s\\n" "$result"'
                    delta={'role':'assistant','tool_calls':[{'index':0,'id':'default_pytest','type':'function','function':{'name':'bash','arguments':json.dumps({'command':cmd})}}]};finish='tool_calls'
                else:delta={'role':'assistant','content':'CPU_PROBE_DONE'};finish='stop'
                self.send_response(200);self.send_header('Content-Type','text/event-stream');self.end_headers()
                for d,f,u in [(delta,None,None),({},finish,None),({},None,{'prompt_tokens':100,'completion_tokens':10,'total_tokens':110})]:
                    obj={'id':'mock','object':'chat.completion.chunk','created':int(time.time()),'model':'Qwen3.5-27B','choices':[] if u else [{'index':0,'delta':d,'finish_reason':f}],'usage':u}
                    self.wfile.write(('data: '+json.dumps(obj)+'\n\n').encode())
                self.wfile.write(b'data: [DONE]\n\n')
        server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
        threading.Thread(target=server.serve_forever,daemon=True).start()
        try:status=execute(case,f'http://127.0.0.1:{server.server_port}/v1','Run the repository default pytest command.')
        finally:server.shutdown();server.server_close()
        raw=(case/'sandbox/artifacts/shell-terminal.raw').read_text(errors='replace')
        checks={'dsh_completed':status['status']=='completed','default_pytest_exit_zero':'DEFAULT_PYTEST_EXIT=0' in raw,'tests_ran':'passed' in raw,'no_unrecognized_options':'unrecognized arguments' not in raw}
        result={'instance_id':instance,'real_model':False,'checks':checks,'passed':all(checks.values()),'case':str(case)}
        write(case/'verification.json',result);results.append(result);print(json.dumps(result),flush=True)
    write(out/'verification.json',{'real_model':False,'passed':all(r['passed'] for r in results),'results':results})
    assert all(r['passed'] for r in results)

if __name__=='__main__':main()
