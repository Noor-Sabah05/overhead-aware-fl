"""Reproducible pilot, shuffled factorial experiment, and CLI. Run from any directory."""
import os
# Set before importing torch/numpy and inherited by every spawned child.
for key in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS']:
    os.environ[key] = '1'
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import random
import subprocess
import sys
import time
import numpy as np
import psutil
import torch
from core import ROOT, atomic_json, sha, initial_state, fedavg, partition, prepare_data, batch_order
from controllers import ReactiveController, ProbeController
from workers import WorkerPool, Interference

METHODS = ['max_fixed', 'pilot_fixed', 'reactive', 'paired_ungated', 'proposed']


def read_config(path):
    cfg = json.loads(Path(path).read_text())
    for k in ['subset', 'clients', 'rounds', 'epochs', 'batch', 'probe_batches', 'probe_every', 'horizon_epochs', 'pilot_examples']:
        if not isinstance(cfg[k], int) or cfg[k] < 1:
            raise ValueError(f'{k} must be a positive integer')
    if cfg['subset'] > 60000 or cfg['clients'] > cfg['subset'] or cfg['alpha'] <= 0:
        raise ValueError('Invalid subset/client/Dirichlet configuration')
    if cfg['lr'] <= 0 or not 0 <= cfg['momentum'] < 1:
        raise ValueError('Invalid SGD parameters')
    if not cfg['threads'] or cfg['threads'] != sorted(set(cfg['threads'])) or min(cfg['threads']) < 1:
        raise ValueError('threads must be sorted, unique, positive')
    if max(cfg['threads']) > (os.cpu_count() or 1):
        raise ValueError('Thread candidates exceed available logical CPUs')
    for k in ['seeds', 'pilot_seeds']:
        if not cfg[k] or len(set(cfg[k])) != len(cfg[k]) or min(cfg[k]) < 0:
            raise ValueError(f'{k} must be unique, nonnegative, nonempty')
    if set(cfg['seeds']) & set(cfg['pilot_seeds']):
        raise ValueError('Pilot and evaluation seeds must be disjoint')
    for k in ['methods', 'conditions', 'pilot_conditions']:
        if not cfg[k] or len(set(cfg[k])) != len(cfg[k]):
            raise ValueError(f'{k} must be nonempty and unique')
    if not set(cfg['methods']) <= set(METHODS):
        raise ValueError('Unknown method')
    if not set(cfg['conditions'] + cfg['pilot_conditions']) <= {'none', 'steady', 'intermittent', 'compute'}:
        raise ValueError('Unknown condition')
    if 'none' not in cfg['pilot_conditions'] or cfg['period_s'] <= 0 or not 0 < cfg['on_s'] <= cfg['period_s']:
        raise ValueError('Invalid pilot or load schedule')
    if not 0 < cfg['min_gain'] < 1 or cfg['payback_factor'] <= 0 or cfg['interference_mb'] <= 0:
        raise ValueError('Invalid controller/load parameters')
    if cfg['interference_mb'] * 3 * 1024**2 > psutil.virtual_memory().available * .25:
        raise ValueError('Interference allocation exceeds 25% of currently available RAM')
    return cfg


def machine_info():
    cpu = platform.processor()
    if sys.platform == 'darwin':
        p = subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'], text=True, capture_output=True)
        if p.returncode == 0:
            cpu = p.stdout.strip()
    elif Path('/proc/cpuinfo').exists():
        for line in Path('/proc/cpuinfo').read_text().splitlines():
            if line.startswith('model name'):
                cpu = line.split(':',1)[1].strip()
                break
    return dict(cpu=cpu or 'unavailable', physical_cores=psutil.cpu_count(logical=False),
        logical_cores=psutil.cpu_count(), ram_gib=psutil.virtual_memory().total / 1024**3,
        platform=platform.platform(), python=platform.python_version(), torch=torch.__version__,
        numpy=np.__version__, affinity='not pinned', device='CPU', hardware_counters=False,
        power_settings='not controlled or recorded', thread_environment={k:os.environ[k] for k in
            ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS']})


def source_hashes():
    return {p.name: sha(p) for p in sorted((ROOT/'src').glob('*.py')) if p.name in
        {'core.py','workers.py','controllers.py','experiment.py'}}


def manifest_for(cfg, dataset):
    payload = dict(config=cfg, dataset=dataset, source_sha256=source_hashes(), environment=machine_info())
    identity = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return {**payload, 'experiment_id': identity, 'created_utc': datetime.now(timezone.utc).isoformat()}


def ensure_manifest(out, manifest):
    path = out / 'manifest.json'
    if path.exists():
        old = json.loads(path.read_text())
        if old['experiment_id'] != manifest['experiment_id']:
            raise ValueError('Config, data, source or machine changed. Use a new --out directory; do not mix runs.')
        return old
    atomic_json(path, manifest)
    return manifest


def pilot(data_path, cfg, out, experiment_id):
    path = out / 'pilot.json'
    if path.exists():
        result = json.loads(path.read_text())
        if result['experiment_id'] != experiment_id:
            raise ValueError('Pilot belongs to another experiment')
        return result
    pool = WorkerPool(data_path, cfg)
    rows = []
    try:
        # Pilot picks ONLY no-load best; load-specific timings are diagnostics.
        for condition in cfg['pilot_conditions']:
            load = Interference(condition, cfg)
            try:
                load.begin()
                for seed in cfg['pilot_seeds']:
                    indices = np.random.default_rng(seed).permutation(cfg['subset'])[:cfg['pilot_examples']]
                    order = list(cfg['threads'])
                    random.Random(seed).shuffle(order)
                    for threads in order:
                        load.check()
                        result = pool.call(threads, 'train', initial_state(seed), indices)
                        rows.append(dict(condition=condition, seed=seed, threads=threads,
                            train_s=result['train_s'], wall_s=result['wall_s'], examples=result['examples']))
                        print(f'pilot {condition:12} seed={seed} threads={threads}: {result["wall_s"]:.3f}s', flush=True)
            finally:
                load.close()
    finally:
        pool.close()
    means = {str(t):float(np.mean([r['wall_s'] for r in rows if r['threads']==t and r['condition']=='none']))
             for t in cfg['threads']}
    throughputs = {str(t):float(np.median([r['examples']/r['train_s'] for r in rows if r['threads']==t and r['condition']=='none']))
                   for t in cfg['threads']}
    result = dict(experiment_id=experiment_id, rows=rows, no_load_wall_s=means,
                  baseline_throughput=throughputs, best_threads=int(min(means, key=means.get)))
    atomic_json(path, result)
    print(f'Pilot selected {result["best_threads"]} threads from independent no-load measurements.', flush=True)
    return result


def run_one(data_path, cfg, pilot_result, seed, method, condition, out, experiment_id):
    directory = out / 'runs' / f'{condition}__{method}__seed{seed}'
    directory.mkdir(parents=True, exist_ok=True)
    summary_path = directory / 'summary.json'
    if summary_path.exists():
        result = json.loads(summary_path.read_text())
        if result['experiment_id'] != experiment_id:
            raise ValueError('Run identity mismatch')
        if sha(directory/'checkpoint.npz') != result['checkpoint_sha256']:
            raise ValueError('Completed checkpoint was modified or damaged')
        return result
    if (directory/'epochs.jsonl').exists():
        # Do not append to incomplete training: restart with exact seed from round zero.
        stamp = str(time.time_ns())
        directory.rename(directory.with_name(directory.name + '.interrupted-' + stamp))
        directory.mkdir()
    with np.load(data_path, allow_pickle=False) as d:
        parts = partition(d['y'], cfg['clients'], cfg['alpha'], seed)
        histogram = [np.bincount(d['y'][idx], minlength=10).tolist() for idx in parts]
    atomic_json(directory/'partitions.json', {'indices':[p.tolist() for p in parts], 'label_histograms':histogram})
    start = time.perf_counter()
    state = initial_state(seed)
    controllers = []
    for _ in parts:
        if method == 'reactive':
            controllers.append(ReactiveController(cfg['threads'], pilot_result['baseline_throughput']))
        elif method in ('paired_ungated','proposed'):
            controllers.append(ProbeController(cfg['threads'], pilot_result['best_threads'], cfg, method=='proposed'))
        else:
            controllers.append(None)
    pool = load = None
    epochs, rounds = [], []
    controller_s = 0.0
    try:
        pool = WorkerPool(data_path, cfg)
        load = Interference(condition, cfg)
        load.begin()
        with open(directory/'epochs.jsonl', 'w') as log:
            for r in range(cfg['rounds']):
                round_start = time.perf_counter()
                locals_ = []
                for c, indices in enumerate(parts):
                    local, optimizer = state, None  # momentum restarts at global round boundary
                    for e in range(cfg['epochs']):
                        load.check()
                        order = batch_order(indices, seed, r, c, e)
                        controller = controllers[c]
                        local_epoch = r*cfg['epochs']+e
                        remaining = cfg['rounds']*cfg['epochs']-local_epoch
                        decision = {'event':'fixed'}
                        tick = time.perf_counter()
                        if method == 'max_fixed':
                            threads = max(cfg['threads'])
                        elif method == 'pilot_fixed':
                            threads = pilot_result['best_threads']
                        elif method == 'reactive':
                            threads = controller.threads
                        else:
                            threads, decision = controller.choose(pool, local, optimizer, order, local_epoch, remaining)
                        if controller is not None:
                            controller_s += time.perf_counter()-tick
                        result = pool.call(threads, 'train', local, order, optimizer)
                        local, optimizer = result.pop('state'), result.pop('optimizer')
                        if method == 'reactive':
                            tick = time.perf_counter()
                            decision = controller.observe(result['examples']/result['train_s'])
                            controller_s += time.perf_counter()-tick
                        row = dict(round=r+1, client=c, epoch=e+1, local_epoch=local_epoch,
                            decision=decision, order_sha256=hashlib.sha256(order.tobytes()).hexdigest(), **result)
                        epochs.append(row)
                        log.write(json.dumps(row, allow_nan=False)+'\n')
                        log.flush()
                    locals_.append(local)
                state = fedavg(locals_, [len(p) for p in parts])
                evaluation = pool.call(min(cfg['threads']), 'eval', state)
                rounds.append(dict(round=r+1, accuracy=evaluation['accuracy'], eval_wall_s=evaluation['wall_s'],
                                   round_wall_s=time.perf_counter()-round_start))
                atomic_json(directory/'rounds.json', rounds)
                print(f'  {condition}/{method}/s{seed} round {r+1}/{cfg["rounds"]} acc={evaluation["accuracy"]:.4f}', flush=True)
        np.savez(directory/'checkpoint.npz', **state)
        checkpoint_sha = sha(directory/'checkpoint.npz')
        load.check()
    finally:
        if load:
            load.close()
        if pool:
            pool.close()
    total = time.perf_counter()-start
    changes = 0
    for c in range(cfg['clients']):
        threads = [e['threads'] for e in epochs if e['client']==c]
        changes += sum(a!=b for a,b in zip(threads, threads[1:]))
    probes = [e['decision'] for e in epochs if e['decision']['event']=='probe']
    examples = sum(e['examples'] for e in epochs)
    assert examples == cfg['subset'] * cfg['rounds'] * cfg['epochs']
    result = dict(experiment_id=experiment_id, method=method, condition=condition, seed=seed,
        total_s=total, train_s=sum(e['train_s'] for e in epochs), epoch_wall_s=sum(e['wall_s'] for e in epochs),
        eval_s=sum(r['eval_wall_s'] for r in rounds), controller_s=controller_s,
        probe_s=sum(p['probe_wall_s'] for p in probes), probe_count=len(probes),
        accepted_probes=sum(p['accepted'] for p in probes), probe_examples=sum(p['probe_examples'] for p in probes),
        thread_changes=changes, initial_threads=epochs[0]['threads'], train_examples=examples,
        examples_per_s=examples/total, final_accuracy=rounds[-1]['accuracy'],
        checkpoint_sha256=checkpoint_sha, rounds=rounds,
        thread_histogram={str(t):sum(e['threads']==t for e in epochs) for t in cfg['threads']})
    atomic_json(summary_path, result)
    print(f'DONE {condition}/{method}/s{seed}: {total:.2f}s, accuracy {result["final_accuracy"]:.4f}, probes {len(probes)}', flush=True)
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('command', choices=['check','prepare','pilot','run','all','analyze'])
    ap.add_argument('--config', type=Path, default=ROOT/'configs/short.json')
    ap.add_argument('--out', type=Path, default=ROOT/'results/short')
    ap.add_argument('--cache', type=Path, default=ROOT/'dataset')
    args=ap.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if args.command=='check':
        print(json.dumps(machine_info(), indent=2))
        return
    if args.command=='analyze':
        from analyze import analyze
        analyze(args.out)
        return
    cfg=read_config(args.config)
    data_path, dataset=prepare_data(cfg,args.cache)
    if args.command=='prepare':
        print(data_path)
        return
    args.out.mkdir(parents=True, exist_ok=True)
    manifest=ensure_manifest(args.out, manifest_for(cfg,dataset))
    pil=pilot(data_path,cfg,args.out,manifest['experiment_id'])
    if args.command=='pilot':
        return
    jobs=[(s,m,c) for s in cfg['seeds'] for c in cfg['conditions'] for m in cfg['methods']]
    random.Random(cfg['schedule_seed']).shuffle(jobs)
    atomic_json(args.out/'schedule.json', jobs)
    for i,(seed,method,condition) in enumerate(jobs):
        print(f'RUN {i+1}/{len(jobs)}: {condition}/{method}/s{seed}',flush=True)
        run_one(data_path,cfg,pil,seed,method,condition,args.out,manifest['experiment_id'])
    if args.command=='all':
        from analyze import analyze
        analyze(args.out)


if __name__=='__main__':
    main()
