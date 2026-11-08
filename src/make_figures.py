"""Original vector architecture, CNN, and decision-flow figures for this project."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT=Path(__file__).resolve().parents[1]/'results/short/analysis'
OUT.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','svg.fonttype':'none'})


def box(ax,x,y,w,h,title,body,color='#EAF1F7'):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.015,rounding_size=0.025',
        fc=color,ec='#526779',lw=1.3))
    ax.text(x+w/2,y+h*.73,title,ha='center',va='center',fontsize=12.5,weight='bold',color='#183449')
    ax.text(x+w/2,y+h*.34,body,ha='center',va='center',fontsize=11,linespacing=1.5,color='#263844')


def arrow(ax,a,b,label=None,offset=(0,.025)):
    ax.annotate('',xy=b,xytext=a,arrowprops=dict(arrowstyle='->',lw=1.6,color='#526779'))
    if label:ax.text((a[0]+b[0])/2+offset[0],(a[1]+b[1])/2+offset[1],label,ha='center',va='center',fontsize=10,color='#526779')


def save(fig,name):
    fig.savefig(OUT/f'{name}.png',dpi=240,bbox_inches='tight',facecolor='white')
    fig.savefig(OUT/f'{name}.svg',bbox_inches='tight',facecolor='white')
    plt.close(fig)


def main():
    fig,ax=plt.subplots(figsize=(12,6.1));ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
    box(ax,.025,.70,.27,.25,'Verified Fashion-MNIST','Fixed subset and test set\nFour non-IID client partitions')
    box(ax,.37,.70,.27,.25,'FedAvg coordinator','Sequential logical clients\nSample-weighted aggregation')
    box(ax,.715,.70,.26,.25,'Evaluation and audit','Accuracy, wall time, hashes\nCheckpoints and epoch logs')
    box(ax,.025,.16,.27,.28,'Background process','None / steady / intermittent\nArray traffic on shared host')
    box(ax,.385,.16,.24,.28,'Persistent worker pool','One worker per 1, 2, 4 threads\nOnly one trains at a time')
    box(ax,.715,.16,.26,.28,'Per-client controller','Matched disposable probes\nGain and overhead gate', '#E7F2EE')
    arrow(ax,(.30,.825),(.36,.825));arrow(ax,(.645,.825),(.705,.825))
    arrow(ax,(.505,.69),(.505,.46),'State + data order',(.10,0))
    arrow(ax,(.30,.30),(.36,.30))
    arrow(ax,(.645,.37),(.705,.37),'Timing',(-.006,.045))
    arrow(ax,(.705,.23),(.645,.23),'Threads',(-.006,-.05))
    ax.text(.5,.05,'Single-host FL simulation  |  CPU kernel parallelism + separate background process  |  No network benchmark',
        ha='center',fontsize=10,color='#526779')
    save(fig,'architecture')
    fig,ax=plt.subplots(figsize=(12,3.1));ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
    nodes=[('Input','1 × 28 × 28\nNormalized FP32'),('Conv + pool','1 → 16, kernel 3\n16 × 13 × 13'),
        ('Conv + pool','16 → 32, kernel 3\n32 × 5 × 5'),('Classifier','Flatten 800\nLinear 800 → 10')]
    for i,(title,body) in enumerate(nodes):
        x=.02+i*.25
        box(ax,x,.28,.205,.52,title,body)
        if i<3:arrow(ax,(x+.21,.54),(x+.24,.54))
    ax.text(.5,.10,'ReLU before each 2 × 2 max pool  |  No convolution padding  |  12,810 trainable parameters',ha='center',fontsize=10)
    save(fig,'cnn')
    fig,ax=plt.subplots(figsize=(11,7));ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
    box(ax,.03,.72,.29,.23,'Epoch boundary','Every fourth local epoch\nOtherwise retain thread count')
    box(ax,.38,.72,.25,.23,'Clone and compare','Same model + momentum\nSame batches; ABBA/BAAB')
    box(ax,.70,.72,.27,.23,'Discard probe updates','Mean time per example\nMeasure full probe wall cost')
    box(ax,.38,.36,.25,.23,'Acceptance rule','Gain ≥ 5% AND\nsaving > 1.25 × probe cost','#E7F2EE')
    box(ax,.03,.05,.29,.22,'Reject or cooldown','Keep threads; train once\nRecord reason')
    box(ax,.70,.05,.27,.22,'Accept candidate','Train the real epoch once\nPreserve SGD work')
    arrow(ax,(.33,.835),(.37,.835));arrow(ax,(.64,.835),(.69,.835))
    arrow(ax,(.835,.71),(.635,.57))
    arrow(ax,(.38,.40),(.25,.28),'No',(-.025,.015))
    arrow(ax,(.63,.40),(.78,.28),'Yes',(.025,.015))
    ax.text(.17,.49,'Only adjacent candidates\nwithin {1, 2, 4}',ha='center',fontsize=10,color='#526779')
    ax.text(.83,.49,'Horizon ≤ 4 epochs\nand remaining work',ha='center',fontsize=10,color='#526779')
    save(fig,'controller')

if __name__=='__main__':main()
