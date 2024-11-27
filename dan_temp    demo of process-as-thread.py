# demo of process-as-thread

# Note: 这段代码只是思考题，不是最后能用的骨架。请勿以它为模板来填写。
# todo: allow the LLM agent to kill the process. hint: pass `process` out from the thread.

import subprocess
import threading

def runGitPullBlocking():
    with subprocess.Popen('git', 'pull') as process:
        while True:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                ... # notify LLM agent of (how much time passed, new stdout stderr)
                # you can use `queue` or `yield`
                continue
        ... # notify LLM agent of (process finished, remaining stdout stderr)
    
thread = threading.Thread(target=runGitPullBlocking)
thread.start()
