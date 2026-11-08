"""Spawn-safe persistent workers. Each worker fixes intra-op threads before tensor work."""
import multiprocessing as mp
import os
import time
import traceback
import numpy as np


def worker_main(conn, threads, data_path, cfg):
    try:
        import torch
        torch.set_num_threads(threads)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        from torch.nn import functional as F
        from core import SmallCNN, to_torch, to_numpy
        with np.load(data_path, allow_pickle=False) as d:
            x = (torch.from_numpy(d['x'].astype(np.float32)) / 255 - .2860) / .3530
            y = torch.from_numpy(d['y'])
            xt = (torch.from_numpy(d['xt'].astype(np.float32)) / 255 - .2860) / .3530
            yt = torch.from_numpy(d['yt'])
        model = SmallCNN()
        opt = torch.optim.SGD(model.parameters(), lr=cfg['lr'], momentum=cfg['momentum'])
        # Identical discarded warm-up for every method and thread worker.
        for _ in range(2):
            opt.zero_grad(set_to_none=True)
            F.cross_entropy(model(x[:cfg['batch']]), y[:cfg['batch']]).backward()
            opt.step()
        conn.send({'ready': True, 'threads': torch.get_num_threads(), 'pid': os.getpid()})
        while True:
            msg = conn.recv()
            if msg['cmd'] == 'close':
                break
            model.load_state_dict(to_torch(msg['state']))
            if msg['cmd'] == 'eval':
                start = time.perf_counter()
                model.eval()
                correct = 0
                with torch.no_grad():
                    for s in range(0, len(yt), 512):
                        correct += (model(xt[s:s+512]).argmax(1) == yt[s:s+512]).sum().item()
                conn.send({'accuracy': correct / len(yt), 'eval_s': time.perf_counter() - start})
                continue
            model.train()
            opt = torch.optim.SGD(model.parameters(), lr=cfg['lr'], momentum=cfg['momentum'])
            if msg.get('optimizer') is not None:
                opt.load_state_dict(to_torch(msg['optimizer']))
            ids = msg['indices']
            times, seen = [], 0
            start = time.perf_counter()
            for s in range(0, len(ids), cfg['batch']):
                b = torch.as_tensor(ids[s:s+cfg['batch']], dtype=torch.long)
                t = time.perf_counter()
                opt.zero_grad(set_to_none=True)
                loss = F.cross_entropy(model(x[b]), y[b])
                loss.backward()
                opt.step()
                times.append(time.perf_counter() - t)
                seen += len(b)
            elapsed = time.perf_counter() - start
            result = {'train_s': elapsed, 'batch_s': times, 'examples': seen,
                      'threads': threads, 'loss_last_batch': float(loss.detach())}
            if msg['cmd'] == 'train':
                result.update(state=to_numpy(model.state_dict()), optimizer=to_numpy(opt.state_dict()))
            # Probe states are deliberately not returned or committed.
            conn.send(result)
    except BaseException:
        try:
            conn.send({'error': traceback.format_exc()})
        except (BrokenPipeError, OSError):
            pass
    finally:
        conn.close()


class WorkerPool:
    def __init__(self, data_path, cfg):
        self.ctx = mp.get_context('spawn')
        self.workers = {}
        try:
            # Sequential startup avoids a transient oversubscription burst.
            for threads in cfg['threads']:
                parent, child = self.ctx.Pipe()
                proc = self.ctx.Process(target=worker_main, args=(child, threads, str(data_path), cfg))
                proc.start()
                child.close()
                self.workers[threads] = (parent, proc)
                self.receive(threads)
        except BaseException:
            self.close()
            raise

    def receive(self, threads, timeout=180):
        conn, proc = self.workers[threads]
        deadline = time.monotonic() + timeout
        while not conn.poll(.1):
            if not proc.is_alive():
                raise RuntimeError(f'Worker {threads} exited: {proc.exitcode}')
            if time.monotonic() > deadline:
                raise TimeoutError(f'Worker {threads} timed out')
        try:
            result = conn.recv()
        except EOFError as exc:
            raise RuntimeError(f'Worker {threads} closed its connection') from exc
        if 'error' in result:
            raise RuntimeError(result['error'])
        return result

    def call(self, threads, cmd, state, indices=None, optimizer=None):
        start = time.perf_counter()
        self.workers[threads][0].send(dict(cmd=cmd, state=state, indices=indices, optimizer=optimizer))
        result = self.receive(threads)
        result['wall_s'] = time.perf_counter() - start
        return result

    def close(self):
        for conn, proc in self.workers.values():
            try:
                if proc.is_alive():
                    conn.send({'cmd': 'close'})
            except (OSError, BrokenPipeError):
                pass
        for conn, proc in self.workers.values():
            proc.join(timeout=5)
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=5)
            conn.close()
        self.workers.clear()


def interference_main(condition, mb, period, on_seconds, stop, ready, start_event):
    # Numpy ufunc loops (not BLAS). Three arrays, each mb MiB.
    if condition == 'compute':
        a = np.ones(2048, dtype=np.float64)
    else:
        n = int(mb * 1024 * 1024 / 8)
        a, b, c = np.ones(n), np.ones(n), np.empty(n)
    ready.set()
    start_event.wait()
    origin = time.monotonic()
    while not stop.is_set():
        if condition == 'compute':
            for _ in range(100):
                np.sin(a, out=a)
                np.cos(a, out=a)
        elif condition == 'steady' or ((time.monotonic() - origin) % period) < on_seconds:
            np.add(a, b, out=c)
            np.multiply(c, .999999, out=a)
        else:
            stop.wait(.005)


class Interference:
    def __init__(self, condition, cfg):
        self.proc = None
        if condition == 'none':
            return
        ctx = mp.get_context('spawn')
        self.stop, ready, self.start_event = ctx.Event(), ctx.Event(), ctx.Event()
        self.proc = ctx.Process(target=interference_main,
            args=(condition, cfg['interference_mb'], cfg['period_s'], cfg['on_s'], self.stop, ready, self.start_event))
        self.proc.start()
        if not ready.wait(30):
            self.close()
            raise RuntimeError('Interference process failed to initialize')

    def begin(self):
        if self.proc:
            self.start_event.set()

    def check(self):
        if self.proc and not self.proc.is_alive():
            raise RuntimeError('Background process died; condition is invalid')

    def close(self):
        if self.proc:
            self.stop.set()
            self.start_event.set()
            self.proc.join(timeout=5)
            if self.proc.is_alive():
                self.proc.terminate()
                self.proc.join(timeout=5)
            self.proc = None
