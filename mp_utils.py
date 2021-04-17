import os
import multiprocessing as mp
import time
import queue
import threading

#from linux_utils import init_rt_stacktrace
#init_rt_stacktrace()


class EntryPoint:
    def __init__(self):
        self.q = mp.Queue()
    def __next__(self):
        return self.q.get()
    def put(self, x):
        self.q.put(x)
    def __iter__(self):
        return self

class _MpGen:
    def __init__(self, inp_it, maxsize=20, engine='mp'):
        self.done_cnt = mp.Value('i', 0)
        self.num_sent = mp.Value('i', 0)
        self.num_recv = mp.Value('i', 0)
        if engine=='mp':
            self.q = mp.Queue(maxsize)
            self.cv = mp.Condition()
            creat_fn = mp.Process
        else:
            self.q = queue.Queue(maxsize)
            self.cv = threading.Condition()
            creat_fn = threading.Thread
        self.p = creat_fn(target=self.feeder, args=(inp_it,))
        self.p.start()

    def join(self):
        self.p.join()

    def __iter__(self):
        return self

    def feeder(self, inp_it):
        try:
            for x in inp_it:
                self.q.put(x)
                with self.cv:
                    self.num_sent.value += 1
        finally:
            with self.cv:
                self.cv.wait_for(lambda: self.num_sent.value == self.num_recv.value)
                self.done_cnt.value += 1
    
    def __next__(self):
        while True:
            try:
                res = self.q.get(True, 1)
                with self.cv:
                    self.num_recv.value += 1
                    self.cv.notify_all()
                return res
            except queue.Empty:
                with self.cv:
                    if self.num_sent.value == self.num_recv.value and self.done_cnt.value == 1:
                        raise StopIteration


class MpGen:
    def __init__(self, inp_it, worker_fn=None, num_workers=1, worker_cls=None, worker_cls_kw={}, streaming_mode=False, maxsize=20, engine='mp'):
        self.pid = os.getpid()
        if engine == 'mp':
            self.q = mp.Queue(maxsize)
        else:
            self.q = queue.Queue(maxsize)
        self.streaming_mode = streaming_mode
        self.done_cnt = mp.Value('i', 0)
        self.num_sent = mp.Value('i', 0)
        self.num_recv = mp.Value('i', 0)
        if engine == 'mp':
            self.cv = mp.Condition()
        else:
            self.cv = threading.Condition()
        self.num_workers = num_workers
        if num_workers > 0:
            if type(inp_it) is MpGen or type(inp_it) is _MpGen:
                self.inp_it = inp_it
            else:
                self.inp_it = _MpGen(inp_it, engine=engine)

            self.procs = []
            for widx in range(num_workers):
                if engine == 'mp':
                    creat_fn = mp.Process
                else:
                    creat_fn = threading.Thread
                p = creat_fn(target=self.worker, args=(widx, self.q, worker_fn, worker_cls, worker_cls_kw))
                p.start()
                self.procs.append(p)
        else:
            self.inp_it = inp_it
            if worker_cls is not None:
                self.worker_fn = worker_cls(rank=0, **worker_cls_kw)
            else:
                self.worker_fn = worker_fn
            def _input_it():
                if self.streaming_mode:
                    for x in self.worker_fn(self.inp_it, rank=0) or []:
                        yield x
                else:
                    for x in self.inp_it:
                        for y in self.worker_fn(x):
                            yield y
            self.input_it = _input_it()

            

    def join(self):
        for p in self.procs:
            p.join()
        self.inp_it.join()

    def worker(self, worker_idx, q, worker_fn=None, worker_cls=None, worker_cls_kw={}):
        if worker_cls is not None:
            worker_fn = worker_cls(rank=worker_idx, **worker_cls_kw)
        try:
            if self.streaming_mode:
                for x in worker_fn(self.inp_it, rank=worker_idx) or []:
                    self.q.put(x)
                    with self.cv:
                        self.num_sent.value += 1
            else:
                for data in self.inp_it:
                    res = worker_fn(data)
                    for x in res or []:
                        self.q.put(x)
                        with self.cv:
                            self.num_sent.value += 1
        finally:
            with self.cv:
                self.cv.wait_for(lambda: self.num_sent.value == self.num_recv.value)
                self.done_cnt.value += 1

    def __next__(self):
        if self.num_workers > 0:
            while True:
                try:
                    res = self.q.get(True, 1)
                    with self.cv:
                        self.num_recv.value += 1
                        self.cv.notify_all()
                    return res
                except queue.Empty:
                    with self.cv:
                        if self.num_sent.value == self.num_recv.value and self.done_cnt.value == self.num_workers:
                            if self.pid == os.getpid():
                                self.join()
                            raise StopIteration
        else:
            return next(self.input_it)



    def __iter__(self):
        return self


if __name__ == "__main__":
    def delay(fn):
        def _fn(*args, **kwargs):
            import random
            t = random.uniform(0,1)
            time.sleep(t)
            return fn(*args, **kwargs)
        return _fn

    def stream(iterable, rank):
        for x in iterable:
            yield x

    def counter(iterable, rank):
        for x in iterable:
            if x == 1:
                raise Exception
            print(x)
        return []
    
    g0 = range(10)
    g1 = MpGen(g0, delay(lambda x: [x*x]), num_workers=0)
    g2 = MpGen(g1, delay(lambda x: [x+1]), num_workers=0)
    g3 = MpGen(g2, stream, streaming_mode=True, num_workers=0)
    for x in g3:
        print(x)
    print('PASS')

    g0 = range(10)
    g1 = MpGen(g0, delay(lambda x: [x*x]), num_workers=3)
    g2 = MpGen(g1, delay(lambda x: [x+1]), num_workers=2)
    g3 = MpGen(g2, stream, streaming_mode=True, num_workers=2)
    g4 = MpGen(g3, counter, streaming_mode=True, num_workers=2)
    for x in g4:
        print(x)
    print('PASS')
