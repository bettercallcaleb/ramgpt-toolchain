import math
import threading
import time

class MemoryTrace:
    def __init__(self,uuid,interval=.1):
        import pynvml as nv
        self.nv=nv
        self.uuid=uuid
        self.interval=interval
        self.samples=[]
        self.errors=[]
        self.stop=threading.Event()

    def sample(self):
        try:
            m=self.nv.nvmlDeviceGetMemoryInfo(self.handle,version=self.nv.nvmlMemory_v2)
            self.samples.append({'elapsed_s':time.monotonic()-self.start,'used_bytes':int(m.used),'total_bytes':int(m.total),'reserved_bytes':int(m.reserved),'free_bytes':int(m.free)})
        except Exception as e:self.errors.append(str(e))

    def __enter__(self):
        self.nv.nvmlInit()
        try:
            self.handle=self.nv.nvmlDeviceGetHandleByUUID(self.uuid)
            self.start=time.monotonic()
            self.sample()
            def loop():
                while not self.stop.wait(self.interval):self.sample()
            self.thread=threading.Thread(target=loop,daemon=True)
            self.thread.start()
            return self
        except BaseException:
            self.nv.nvmlShutdown()
            raise

    def __exit__(self,*args):
        self.stop.set()
        self.thread.join()
        self.sample()
        self.nv.nvmlShutdown()

    def report(self):
        if self.errors or len(self.samples)<2:raise ValueError('Incomplete GPU memory telemetry: '+str(self.errors))
        totals={x['total_bytes'] for x in self.samples}
        if len(totals)!=1:raise ValueError('GPU capacity changed')
        used=[x['used_bytes'] for x in self.samples]
        return {'method':'NVML device-wide sampled high-water mark; lower bound on instantaneous peak','interval_s':self.interval,'samples':self.samples,'sampled_peak_bytes':max(used),'baseline_bytes':used[0],'total_bytes':totals.pop(),'minimum_free_bytes':min(x['free_bytes'] for x in self.samples),'max_sample_gap_s':max(b['elapsed_s']-a['elapsed_s'] for a,b in zip(self.samples,self.samples[1:]))}

def timings(raw):
    t=raw['timings'];u=raw['usage']
    for k in ('prompt_n','predicted_n','cache_n'):
        if type(t[k]) is not int or t[k]<0:raise ValueError('Invalid timing counts')
    if t['predicted_n']!=u['completion_tokens'] or t['prompt_n']+t['cache_n']!=u['prompt_tokens']:raise ValueError('Timing and usage mismatch')
    if t['cache_n']!=0:raise ValueError('Unexpected cached prompt')
    for k in ('prompt_ms','predicted_ms'):
        if type(t[k]) not in (float,int) or not math.isfinite(t[k]) or t[k]<=0:raise ValueError('Invalid timing duration')
    return {k:t[k] for k in ('prompt_n','predicted_n','prompt_ms','predicted_ms')}

def summarize(rows):
    ts=[timings(x['raw']) for x in rows]
    if not ts:raise ValueError('Empty timing evidence')
    return {'requests':len(ts),'prefill_tokens_per_s':1000*sum(x['prompt_n'] for x in ts)/sum(x['prompt_ms'] for x in ts),'decode_tokens_per_s':1000*sum(x['predicted_n'] for x in ts)/sum(x['predicted_ms'] for x in ts),'mean_http_latency_s':sum(x['elapsed_s'] for x in rows)/len(rows)}
