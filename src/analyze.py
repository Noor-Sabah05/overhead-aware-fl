"""Tables and publication-quality figures generated only from completed run records."""
from pathlib import Path
import csv
import json
import math
import statistics
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from core import atomic_json, sha

LABELS={'max_fixed':'Max 4', 'pilot_fixed':'Pilot', 'reactive':'Reactive',
        'paired_ungated':'Ungated','proposed':'Proposed'}
COLORS={'max_fixed':'#8B9199','pilot_fixed':'#205C8B','reactive':'#DD9036',
        'paired_ungated':'#9766AC','proposed':'#1A8675'}
plt.rcParams.update({'font.size':14,'axes.spines.top':False,'axes.spines.right':False,
    'axes.titlesize':14,'figure.facecolor':'white','savefig.facecolor':'white','svg.fonttype':'none'})


def write_csv(path, rows):
    if not rows:
        return
    with open(path,'w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sd(v):
    return statistics.stdev(v) if len(v)>1 else None


def save_figure(fig,path):
    fig.savefig(str(path)+'.png',dpi=220,bbox_inches='tight')
    fig.savefig(str(path)+'.svg',bbox_inches='tight')
    plt.close(fig)


def analyze(out):
    out=Path(out)
    manifest=json.loads((out/'manifest.json').read_text())
    cfg=manifest['config']
    pil=json.loads((out/'pilot.json').read_text())
    paths=[p for p in (out/'runs').glob('*/summary.json') if '.interrupted-' not in p.parent.name]
    rows=[json.loads(p.read_text()) for p in paths]
    expected={(c,m,s) for c in cfg['conditions'] for m in cfg['methods'] for s in cfg['seeds']}
    actual=[(r['condition'],r['method'],r['seed']) for r in rows]
    if len(set(actual))!=len(actual) or not set(actual)<=expected:
        raise ValueError('Duplicate or unexpected results')
    if any(r['experiment_id']!=manifest['experiment_id'] for r in rows):
        raise ValueError('Mixed experiment identities')
    if not rows:
        raise ValueError('No completed runs')
    # Verify paired sample ordering and equal training work before drawing conclusions.
    work={r['train_examples'] for r in rows}
    if work!={cfg['subset']*cfg['rounds']*cfg['epochs']}:
        raise ValueError('Training work differs across methods')
    epoch_logs={}
    for path,r in zip(paths,rows):
        log=[json.loads(l) for l in (path.parent/'epochs.jsonl').read_text().splitlines()]
        epoch_logs[(r['condition'],r['method'],r['seed'])]=log
        r['payback_rejections']=sum(e['decision'].get('reason')=='not_amortized' for e in log)
        r['gain_rejections']=sum(e['decision'].get('reason')=='insufficient_gain' for e in log)
    for c in cfg['conditions']:
        for seed in cfg['seeds']:
            orders=[[(e['round'],e['client'],e['epoch'],e['order_sha256']) for e in log]
                    for (cc,_,s),log in epoch_logs.items() if cc==c and s==seed]
            if orders and any(v!=orders[0] for v in orders[1:]):
                raise ValueError('Paired methods used different batch orders')
    dest=out/'analysis'
    dest.mkdir(exist_ok=True)
    missing=sorted(expected-set(actual))
    atomic_json(dest/'completeness.json',{'complete':not missing,'completed':len(rows),'expected':len(expected),'missing':missing})
    scalar_keys=['condition','method','seed','total_s','train_s','epoch_wall_s','eval_s','probe_s','controller_s',
                 'final_accuracy','examples_per_s','train_examples','thread_changes','probe_count','accepted_probes','probe_examples','payback_rejections','gain_rejections']
    write_csv(dest/'per_seed.csv',[{k:r[k] for k in scalar_keys} for r in sorted(rows,key=lambda r:(r['condition'],r['method'],r['seed']))])
    summaries=[]
    for c in cfg['conditions']:
        for m in cfg['methods']:
            group=[r for r in rows if r['condition']==c and r['method']==m]
            if not group:continue
            summary={'condition':c,'method':m,'n':len(group)}
            for key in ['total_s','final_accuracy','train_s','probe_s','controller_s','thread_changes','accepted_probes','examples_per_s','payback_rejections','gain_rejections']:
                values=[r[key] for r in group]
                summary[key+'_mean']=statistics.mean(values)
                summary[key+'_sd']=sd(values)
            summaries.append(summary)
    write_csv(dest/'summary.csv',summaries)
    lookup={(r['condition'],r['method'],r['seed']):r for r in rows}
    paired=[]
    for c in cfg['conditions']:
        for baseline in ['max_fixed','pilot_fixed','reactive','paired_ungated']:
            if baseline not in cfg['methods']:continue
            for method in ['proposed']:
                pairs=[(lookup[(c,baseline,s)],lookup[(c,method,s)]) for s in cfg['seeds']
                       if (c,baseline,s) in lookup and (c,method,s) in lookup]
                if not pairs:continue
                reductions=[100*(1-p['total_s']/b['total_s']) for b,p in pairs]
                speedups=[b['total_s']/p['total_s'] for b,p in pairs]
                deltas=[100*(p['final_accuracy']-b['final_accuracy']) for b,p in pairs]
                paired.append(dict(condition=c,baseline=baseline,method=method,n=len(pairs),
                    time_reduction_pct_mean=statistics.mean(reductions),time_reduction_pct_sd=sd(reductions),
                    speedup_mean=statistics.mean(speedups),accuracy_delta_pp_mean=statistics.mean(deltas),
                    accuracy_delta_pp_sd=sd(deltas),accuracy_gate=statistics.mean(deltas)>=-1,
                    all_seeds_faster=all(x>0 for x in reductions),
                    useful_descriptive=statistics.mean(reductions)>0 and statistics.mean(deltas)>=-1))
    write_csv(dest/'paired_comparisons.csv',paired)
    atomic_json(dest/'analysis.json',{'complete':not missing,'summary':summaries,'paired':paired})
    # Runtime and accuracy retain per-seed observations; SD uses ddof=1.
    fig,axes=plt.subplots(1,len(cfg['conditions']),figsize=(12,3.8),squeeze=False)
    for ax,c in zip(axes[0],cfg['conditions']):
        for i,m in enumerate(cfg['methods']):
            values=[r['total_s'] for r in rows if r['condition']==c and r['method']==m]
            if values:
                ax.bar(i,np.mean(values),yerr=sd(values) or 0,color=COLORS[m],capsize=3,width=.7)
                ax.scatter(np.linspace(i-.12,i+.12,len(values)),values,color='#263238',s=12,zorder=3)
        ax.set_xticks(range(len(cfg['methods'])),[LABELS[m] for m in cfg['methods']],rotation=35,ha='right')
        ax.set_title(c.capitalize())
        ax.set_ylabel('End-to-end time (s)')
    fig.tight_layout()
    save_figure(fig,dest/'runtime')
    fig,axes=plt.subplots(1,len(cfg['conditions']),figsize=(12,3.3),squeeze=False,sharey=True)
    for ax,c in zip(axes[0],cfg['conditions']):
        for m in cfg['methods']:
            group=[r for r in rows if r['condition']==c and r['method']==m]
            if group:
                values=np.array([[rd['accuracy']*100 for rd in r['rounds']] for r in group])
                x=np.arange(1,values.shape[1]+1)
                ax.plot(x,values.mean(0),label=LABELS[m],color=COLORS[m],linewidth=1.7,
                        linestyle='--' if m=='proposed' else '-')
        ax.set_title(c.capitalize());ax.set_xlabel('Global round');ax.set_ylabel('Test accuracy (%)')
    axes[0][-1].legend(fontsize=11)
    fig.tight_layout();save_figure(fig,dest/'accuracy')
    fig,axes=plt.subplots(1,len(cfg['pilot_conditions']),figsize=(10,3.2),squeeze=False)
    for ax,c in zip(axes[0],cfg['pilot_conditions']):
        for i,t in enumerate(cfg['threads']):
            vals=[r['wall_s'] for r in pil['rows'] if r['condition']==c and r['threads']==t]
            if vals:
                ax.bar(i,np.mean(vals),yerr=sd(vals) or 0,color='#205C8B',capsize=3)
                ax.scatter(np.linspace(i-.1,i+.1,len(vals)),vals,c='#263238',s=12)
        ax.set_xticks(range(len(cfg['threads'])),cfg['threads']);ax.set_xlabel('CPU threads')
        ax.set_ylabel('Pilot epoch wall time (s)');ax.set_title(c.capitalize())
    fig.tight_layout();save_figure(fig,dest/'pilot_threads')
    seed=cfg['seeds'][0]
    fig,axes=plt.subplots(1,len(cfg['conditions']),figsize=(12,3.1),squeeze=False,sharey=True)
    for ax,c in zip(axes[0],cfg['conditions']):
        log=epoch_logs.get((c,'proposed',seed),[])
        for client in range(cfg['clients']):
            ep=[e for e in log if e['client']==client]
            if ep:ax.step([e['local_epoch']+1 for e in ep],[e['threads'] for e in ep],where='mid',label=f'Client {client}',alpha=.8)
        ax.set_title(f'{c.capitalize()}, seed {seed}');ax.set_xlabel('Local epoch');ax.set_yticks(cfg['threads'])
    axes[0][0].set_ylabel('Proposed thread count');axes[0][-1].legend(fontsize=11)
    fig.tight_layout();save_figure(fig,dest/'thread_trace')
    fig,axes=plt.subplots(1,len(cfg['conditions']),figsize=(12,3.1),squeeze=False,sharey=True)
    for ax,c in zip(axes[0],cfg['conditions']):
        for m in ['reactive','paired_ungated','proposed']:
            ep=[e for e in epoch_logs.get((c,m,seed),[]) if e['client']==0]
            if ep:
                ax.step([e['local_epoch']+1 for e in ep],[e['threads'] for e in ep],where='mid',
                        label=LABELS[m],color=COLORS[m],linestyle='--' if m=='proposed' else '-',linewidth=1.8)
        ax.set_title(f'{c.capitalize()}, seed {seed}, client 0')
        ax.set_xlabel('Local epoch');ax.set_yticks(cfg['threads'])
    axes[0][0].set_ylabel('Thread count');axes[0][-1].legend(fontsize=11)
    fig.tight_layout();save_figure(fig,dest/'controller_comparison')
    # Partition figure uses the exact run's saved histogram, not regenerated guesses.
    first=paths[0].parent
    hist=json.loads((first/'partitions.json').read_text())['label_histograms']
    fig,ax=plt.subplots(figsize=(8,3))
    im=ax.imshow(hist,cmap='Blues',aspect='auto')
    ax.set_xticks(range(10),['T-shirt','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Boot'],rotation=30,ha='right',fontsize=11)
    ax.set_yticks(range(cfg['clients']),[f'Client {i}' for i in range(cfg['clients'])])
    ax.set_title(f'Dirichlet partition, seed {rows[0]["seed"]}, alpha {cfg["alpha"]}')
    for i in range(cfg['clients']):
        for j in range(10):ax.text(j,i,str(hist[i][j]),ha='center',va='center',fontsize=8,color='white' if hist[i][j]>np.max(hist)*.55 else '#1F2933')
    fig.colorbar(im,ax=ax,label='Training examples')
    fig.tight_layout();save_figure(fig,dest/'partition')
    atomic_json(dest/'analysis_provenance.json', {
        'experiment_id':manifest['experiment_id'], 'analysis_source_sha256':sha(Path(__file__)),
        'summary_sha256':{str(p.relative_to(out)):sha(p) for p in paths},
        'epochs_sha256':{str((p.parent/'epochs.jsonl').relative_to(out)):sha(p.parent/'epochs.jsonl') for p in paths},
        'paired_batch_orders_verified':True,'equal_real_training_work_verified':True})
    print(f'Analysis saved: {dest} ({len(rows)}/{len(expected)} completed runs)',flush=True)
    return summaries,paired
