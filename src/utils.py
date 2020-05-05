import matplotlib.pyplot as plt

def barplot(size,index,values,**kwargs):
    plt.rc('font', size=20) 
    plt.rc('font', serif='Arial') 
    fig = plt.figure(figsize=size)
    ax = fig.subplots()
    ax.set_facecolor('w')
    for spine in ["left", "top", "right","bottom"]:
        ax.spines[spine].set_linewidth(2)  
    for i in ax.xaxis.get_ticklines():
        i.set_markersize(6)
        i.set_markeredgewidth(2)
    for i in ax.yaxis.get_ticklines():
        i.set_markersize(6)
        i.set_markeredgewidth(2)

    ax.barh([str(i) for i in index],values,**kwargs)
    return fig,ax


def corpusGenerator():
    pass

