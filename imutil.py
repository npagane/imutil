import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
import scipy
from scipy import stats
import math
import seaborn as sns
import statannot
from statannotations.Annotator import Annotator
import matplotlib.cm as cm
from loess import loess_1d
from scipy.stats import bootstrap
from matplotlib import colors
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.stats import spearmanr
from scipy.cluster.hierarchy import linkage 
from matplotlib.patches import Patch
import matplotlib
import matplotlib.colors as mcolors
import umap.umap_ as umap
import umap.plot

def read_in_imaris_folder(filename, markers, cell, sample, experiment=None, treatment=None, current_os='\\'):
    dat = pd.DataFrame()
    # add extra expected name to filepath
    pref = filename.split(current_os)[-1].split('_Statistics')[0]
    # read in xyz data
    dat_x = pd.read_csv(filename.rstrip()+current_os+pref+"_Position_X.csv", header = 2)
    dat_y = pd.read_csv(filename.rstrip()+current_os+pref+"_Position_Y.csv", header = 2)
    dat_z = pd.read_csv(filename.rstrip()+current_os+pref+"_Position_Z.csv", header = 2)
    dat['x'] = dat_x["Position X"]
    dat['y'] = dat_y["Position Y"]
    dat['z'] = dat_z["Position Z"]
    # read in marker data
    for i in range(len(markers)):
        temp_mean = pd.read_csv(filename.rstrip()+current_os+pref+"_Intensity_Mean_Ch=" + str(i+1) + "_Img=1.csv", header = 2)
        temp_sum = pd.read_csv(filename.rstrip()+current_os+pref+"_Intensity_Sum_Ch=" + str(i+1) + "_Img=1.csv", header = 2)
        dat[markers[i] + " MFI"] = temp_mean["Intensity Mean"]
        dat[markers[i] + " SFI"] = temp_sum["Intensity Sum"]
    # fill in identifiers for the data
    dat['cell'] = cell
    if treatment == None and experiment == None:
        dat['sample'] = sample
    elif treatment == None and experiment != None:
        dat['sample'] = experiment + "." + sample
        dat['experiment'] = experiment
    elif treatment != None and experiment == None:
        dat['sample'] = treatment + "." + sample
        dat['treatment'] = treatment
    else:
        dat['sample'] = treatment + "." + experiment + "." + sample
        dat['experiment'] = experiment
        dat['treatment'] = treatment
    return dat

def define_blocks(df, blocks):
    df["block"] = -1
    for i, block in enumerate(blocks):
        for j in range(len(block)):
            df.loc[df["sample"] == block[j], 'block'] = i
    return df

def plot_LN(df, sample, cell_list, figsize=(15,15), fontsize=30):
    plt.figure(figsize=figsize)
    ax = plt.axes(projection="3d")
    tempdf = df.loc[df['sample']==sample,]
    # plot points
    temp_scale = np.zeros(len(cell_list))
    for i, cell in enumerate(cell_list):
        temp_scale[i] = np.sum(tempdf['cell']==cell)/len(tempdf)
        ax.scatter3D(tempdf.loc[tempdf['cell']==cell,'x'],tempdf.loc[tempdf['cell']==cell,'y'],tempdf.loc[tempdf['cell']==cell,'z'], label=cell, alpha = 1-0.75*temp_scale[i], s = 50*(1-0.75*temp_scale[i]))
    # adjust plot viewer
    plt.xlabel("x"); plt.ylabel("y"); ax.set_zlabel("z")
    plt.legend(fontsize=2/3*fontsize)
    scale = 1200
    ax.set_xlim([np.mean(tempdf.loc[tempdf['cell']==cell_list[np.argmax(temp_scale)],'x'])-scale, np.mean(tempdf.loc[tempdf['cell']==cell_list[np.argmax(temp_scale)],'x'])+scale])
    ax.set_ylim([np.mean(tempdf.loc[tempdf['cell']==cell_list[np.argmax(temp_scale)],'y'])-scale, np.mean(tempdf.loc[tempdf['cell']==cell_list[np.argmax(temp_scale)],'y'])+scale])
    ax.set_zlim([np.mean(tempdf.loc[tempdf['cell']==cell_list[np.argmax(temp_scale)],'z'])-scale, np.mean(tempdf.loc[tempdf['cell']==cell_list[np.argmax(temp_scale)],'z'])+scale])
    ax.view_init(100, -90)
    ax.set_title(sample, fontsize = fontsize, y = 0.99)
    pass


def plot_KDE(df, sample, cell, cell_list=[], figsize=(15,7), fontsize=30, dx=128, dy=128, fcrit=0.1, bw_method=0.1):
    tempdf = df.loc[df['sample']==sample,]
    # Peform the kernel density estimate
    kernel = scipy.stats.gaussian_kde(tempdf.loc[tempdf['cell']==cell,['x', 'y']].T, bw_method=bw_method)
    # Regular grid to evaluate kde upon
    x_flat = np.r_[tempdf.loc[tempdf['cell']==cell,'x'].min():tempdf.loc[tempdf['cell']==cell,'x'].max():dx*1j]
    y_flat = np.r_[tempdf.loc[tempdf['cell']==cell,'y'].min():tempdf.loc[tempdf['cell']==cell,'y'].max():dy*1j]
    # create meshgrid
    x,y = np.meshgrid(x_flat,y_flat)
    grid_coords_iso = np.append(x.reshape(-1,1),y.reshape(-1,1),axis=1)
    # determine kernerl density
    iso_h = kernel(grid_coords_iso.T) # change kernel size | maybe adaptive kernels?
    temp = (np.abs(x_flat[-1]-x_flat[0])*np.abs(y_flat[-1]-y_flat[0]))/(dx*dy)
    h_reshaped = iso_h.reshape(dx,dy)*temp
    # plot KDE 
    plt.figure(figsize=(figsize[0]*np.abs(np.max(x_flat)-np.min(x_flat))/2000,figsize[1]*np.abs(np.max(y_flat)-np.min(y_flat))/2000))
    plt.grid(False)
    plt.imshow(h_reshaped,aspect=y_flat.ptp()/x_flat.ptp(),origin="lower", cmap='viridis') # vmax? vmin? TODO: figure out correct scaling of KDE 
    xticks = np.linspace(0, np.max(x_flat)-np.min(x_flat),10, dtype="int")
    plt.gca().set_xticklabels(xticks)
    yticks = np.linspace(0, np.max(y_flat)-np.min(y_flat),10, dtype="int")
    plt.gca().set_yticklabels(yticks)
    plt.colorbar(format='%.2e')
    # plot scatter overlay
    cmap = cm.get_cmap(name='Spectral')
    for i,cell in enumerate(cell_list):
        ind = i/len(cell_list)
        for j in range(len(tempdf.loc[tempdf['cell']==cell,])):
            plt.scatter(np.arange(dx)[np.argmin(np.abs(x_flat-tempdf.loc[tempdf['cell']==cell,"x"].iloc[j]))], 
                        np.arange(dy)[np.argmin(np.abs(y_flat-tempdf.loc[tempdf['cell']==cell,"y"].iloc[j]))], 
                        color = cmap(ind), s = 50*(1-0.75*np.sum(tempdf['cell']==cell)/len(tempdf)))
    # pass plot
    plt.title(sample, fontsize = fontsize)
    pass
    # return data frame with covered area percentage 
    df.loc[df['sample']==sample,'2D_KDE%'] = np.sum(h_reshaped >= fcrit/(dx*dy))/(dx*dy)
    return df

def normalize_FI(df, markers, cell_list=[], standard = {"treatment": "WT"}, fi_cutoff=100):
    # iterate until all resonable cells have been normalized and filtered
    bad_ind_sums = 1
    while (bad_ind_sums > 0):
        # determine if normalzing across certain cell types or across all cells
        if len(cell_list) > 0:
            cell_inds = np.zeros(len(df), dtype='bool')
            for cell in cell_list: cell_inds = cell_inds + np.asarray(df['cell'] == cell)
            tempname = " (" + " + ".join(cell_list) + ")"
        else:
            cell_inds = np.zeros(len(df), dtype='bool')
            tempname = " (all)"
        # standardize + filer values for specificed cell type(s)
        for i in range(len(markers)): 
            df[markers[i] + " MFI norm" + tempname] = np.nan
            df[markers[i] + " SFI norm" + tempname] = np.nan
            for block in df.loc[cell_inds,'block'].unique():
                block_inds = np.array(df.loc[cell_inds,'block']==block)
                for key in standard.keys(): block_inds = block_inds * np.asarray(df.loc[cell_inds,key] == standard[key])
                blockMFI = np.mean(df.loc[cell_inds,:].loc[block_inds, markers[i] + " MFI"])
                df.loc[df['block']==block,markers[i] + " MFI norm" + tempname] = np.array(df.loc[np.array(df['block']==block),markers[i] + " MFI"]/blockMFI)
                blockSFI = np.mean(df.loc[cell_inds,:].loc[block_inds, markers[i] + " SFI"])
                df.loc[df['block']==block,markers[i] + " SFI norm" + tempname] = np.array(df.loc[np.array(df['block']==block),markers[i] + " SFI"]/blockSFI)
        # find cells with artificially high SFIs
        keepinds = df[markers[0] + " SFI norm" + tempname] >= fi_cutoff
        for i in range(1,len(markers)):
            keepinds = keepinds | (df[markers[i] + " SFI norm" + tempname]>=fi_cutoff)
        # find cells with artificially high MFIs    
        for i in range(len(markers)):
            keepinds = keepinds | (df[markers[i] + " MFI norm" + tempname]>=fi_cutoff)
        # keep cells with reasonable FIs
        keepinds = ~(keepinds)
        bad_ind_sums = np.sum(~keepinds)
        if (bad_ind_sums > 0):
            df = df.drop(df.index[~keepinds]) 
    return df

def determine_tissue_density(df, ):
    

