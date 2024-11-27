'''
I pushed this file as a reference answer. Only view this after you tried to make one yourself.
仅供阅读、参考。
'''

import typing as tp
import threading
import subprocess
import io
import contextlib
import os
import time
import queue as q

class ProcessAsThread:
    def __init__(
        self, args: tp.List[str],
    ):
        self.last_polled = time.time()
        self.process = subprocess.Popen(
            args, 
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        self.stdout_queue: q.Queue[str | None] = q.Queue()
        self.stderr_queue: q.Queue[str | None] = q.Queue()
        self.outReader = threading.Thread(
            target=self.keepReading, 
            args=(self.process.stdout, self.stdout_queue),
        )
        self.errReader = threading.Thread(
            target=self.keepReading,
            args=(self.process.stderr, self.stderr_queue),
        )
        self.outReader.start()
        self.errReader.start()
    
    @staticmethod
    def keepReading(pipe: io.TextIOBase, queue: q.Queue):
        '''
        I thought this kind of wrapping was unecessary, as I told you.  
        However, I couldn't figure out how to read a pipe in a nonblocking way.
        Hence this.
        '''
        while True:
            char = pipe.read(1)
            if not char:
                break
            queue.put(char)
        queue.put(None)
    
    @staticmethod
    def queueToText(queue: q.Queue):
        buf = io.StringIO()
        while True:
            try:
                char = queue.get_nowait()
            except q.Empty:
                break
            if char is None:
                break
            buf.write(char)
        return buf.getvalue()
    
    def abort(self):    # = Ctrl+C
        self.process.terminate()
    
    def poll(self):
        '''
        `return_code` is None if the process is still running.
        '''
        stdout = self.queueToText(self.stdout_queue)
        stderr = self.queueToText(self.stderr_queue)
        time_elapsed = time.time() - self.last_polled
        self.last_polled += time_elapsed
        return_code = self.process.poll()
        if return_code is not None:
            self.outReader.join()
            self.errReader.join()
        return return_code, time_elapsed, stdout, stderr
    
    @contextlib.contextmanager
    def context(self):
        try:
            yield self.abort, self.poll
        finally:
            self.abort()
            self.process.wait()
            self.outReader.join()
            self.errReader.join()

def test(args: tp.List[str]):
    print('We are about to run')
    print(args)
    input('Press Enter to continue...')
    # os.chdir(os.path.expanduser('~/temp/'))
    # os.mkdir('temp_repo')
    # os.chdir('temp_repo')
    # os.system('git init')
    # os.system('git config remote.origin.url https://github.com/Daniel-Chin/Repo-eval-by-LLM-agent')
    with ProcessAsThread(
        # ['git', 'pull'], 
        args, 
    ).context() as (abort, poll):
        def checkOnce():
            return_code, time_elapsed, stdout, stderr = poll()
            if return_code is None:
                print(f'Process is still running! {time_elapsed:.2f} seconds passed.')
            else:
                print(f'Process finished with return code {return_code} after {time_elapsed:.2f} seconds.')
            if stdout:
                print('  New stdout:')
                print('   ', repr(stdout))
            if stderr:
                print('  New stderr:')
                print('   ', repr(stderr))
            return return_code is None
        
        for _ in range(30):
            time.sleep(0.4)
            if checkOnce():
                continue
            break
        abort() # You can do this at any time
    print()
    print()

if __name__ == '__main__':
    test(['pip', 'uninstall', '--yes', 'scipy'])
    test(['pip', 'uninstall', '--yes', 'numpy'])
    test(['pip', 'cache', 'purge'])
    test(['pip', 'install', 'scipy'])
