import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy
import matplotlib.cm as cm



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

def remove_duplicate_cells(df, subgroup_cell, majorgroup_cell):
    df.index = np.arange(len(df))
    tempdf1 = df.loc[df['cell']==subgroup_cell,]
    for i in range(len(tempdf1)):
        samex = tempdf1.iloc[i]['x'] == df['x']
        samey = tempdf1.iloc[i]['y'] == df['y']
        samez = tempdf1.iloc[i]['z'] == df['z']
        combined = samex*samey*samez
        if np.sum(combined) > 1:
            tempdf2 = df.loc[combined,] 
            badinds = np.asarray(tempdf2['cell']== majorgroup_cell) * np.asarray(tempdf1.iloc[i]['sample'] == tempdf2['sample'])
            df = df.drop(tempdf2.index[badinds]) 
            
    return df

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


def plot_KDE(df, sample, cell_list, scatter_cell_list=[], figsize=(15,7), fontsize=30, dx=128, dy=128, fcrit=0.1, bw_method=0.1, plot=True):
    # make column for 2D KDE percentage if not existent
    if "tissue_2D_KDE%" not in df.columns:
        df["tissue_2D_KDE%"] = np.nan
    # make subset of dataframe to only handle given sample
    tempdf = df.loc[df['sample']==sample,]
    cell_inds = np.zeros(len(tempdf), dtype='bool')
    for cell in cell_list: cell_inds = cell_inds + np.asarray(tempdf['cell'] == cell)
    tempname = " (" + " + ".join(cell_list) + ")"
    # Peform the kernel density estimate
    kernel = scipy.stats.gaussian_kde(tempdf.loc[cell_inds,['x', 'y']].T, bw_method=bw_method)
    # Regular grid to evaluate kde upon
    x_flat = np.r_[tempdf.loc[cell_inds,'x'].min():tempdf.loc[cell_inds,'x'].max():dx*1j]
    y_flat = np.r_[tempdf.loc[cell_inds,'y'].min():tempdf.loc[cell_inds,'y'].max():dy*1j]
    # create meshgrid
    x,y = np.meshgrid(x_flat,y_flat)
    grid_coords_iso = np.append(x.reshape(-1,1),y.reshape(-1,1),axis=1)
    # determine kernerl density
    iso_h = kernel(grid_coords_iso.T) # change kernel size | maybe adaptive kernels?
    temp = (np.abs(x_flat[-1]-x_flat[0])*np.abs(y_flat[-1]-y_flat[0]))/(dx*dy)
    h_reshaped = iso_h.reshape(dx,dy)*temp
    # plot KDE 
    if plot:
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
        for i,cell in enumerate(scatter_cell_list):
            ind = i/len(scatter_cell_list)
            for j in range(len(tempdf.loc[tempdf['cell']==cell,])):
                plt.scatter(np.arange(dx)[np.argmin(np.abs(x_flat-tempdf.loc[tempdf['cell']==cell,"x"].iloc[j]))], 
                            np.arange(dy)[np.argmin(np.abs(y_flat-tempdf.loc[tempdf['cell']==cell,"y"].iloc[j]))], 
                            color = cmap(ind), s = 50*(1-0.75*np.sum(tempdf['cell']==cell)/len(tempdf)))
        # pass plot
        plt.title(sample + tempname + " 2D KDE", fontsize = fontsize)
        pass
    # return data frame with covered area percentage 
    df.loc[df['sample']==sample,"tissue_2D_KDE%"] = np.sum(h_reshaped >= fcrit/(dx*dy))/(dx*dy)
    return df

def normalize_FI(df, markers, cell_list=[], standard = {"treatment": "WT"}, fi_cutoff=100):
    if len(cell_list) > 0:
        tempname = " (" + " + ".join(cell_list) + ")"
    else:
        tempname = " (all)"
    for i in range(len(markers)):
        df[markers[i] + " MFI norm" + tempname] = np.nan
        df[markers[i] + " SFI norm" + tempname] = np.nan
    # iterate until all resonable cells have been normalized and filtered
    bad_ind_sums = 1
    while (bad_ind_sums > 0):
        df.index = np.arange(len(df))
        # determine if normalzing across certain cell types or across all cells
        if len(cell_list) > 0:
            cell_inds = np.zeros(len(df), dtype='bool')
            for cell in cell_list: cell_inds = cell_inds + np.asarray(df['cell'] == cell)
        else:
            cell_inds = np.ones(len(df), dtype='bool')
        # standardize + filer values for specificed cell type(s)
        for i in range(len(markers)): 
            for block in df.loc[cell_inds,'block'].unique():
                block_inds = np.array(df['block']==block)
                standard_inds = np.array(df['block']==block)
                for key in standard.keys(): standard_inds = standard_inds * np.asarray(df[key] == standard[key])
                blockMFI = np.mean(df.loc[cell_inds * standard_inds, markers[i] + " MFI"])
                df.loc[cell_inds * block_inds, markers[i] + " MFI norm" + tempname] = np.array(df.loc[cell_inds * block_inds, markers[i] + " MFI"]/blockMFI)
                blockSFI = np.mean(df.loc[cell_inds * standard_inds, markers[i] + " SFI"])
                df.loc[cell_inds * block_inds, markers[i] + " SFI norm" + tempname] = np.array(df.loc[cell_inds * block_inds, markers[i] + " SFI"]/blockSFI)
        # find cells with artificially high SFIs
        badinds = df[markers[0] + " SFI norm" + tempname] >= fi_cutoff
        for i in range(1,len(markers)):
            badinds = badinds | (df[markers[i] + " SFI norm" + tempname]>=fi_cutoff)
        # find cells with artificially high MFIs    
        for i in range(len(markers)):
            badinds = badinds | (df[markers[i] + " MFI norm" + tempname]>=fi_cutoff)
        # remove cells with artificially high FIs
        bad_ind_sums = np.sum(badinds)
        if (bad_ind_sums > 0):
            df = df.drop(df.index[badinds]) 
    return df

def determine_tissue_geometry(df, cell_list):
    cell_inds = np.zeros(len(df), dtype='bool')
    for cell in cell_list: cell_inds = cell_inds + np.asarray(df['cell'] == cell)
    # run the 2D KDE function if not run yet
    if "tissue_2D_KDE%" not in df.columns:
        for s in df['sample'].unique():
            plot_KDE(df, s, cell_list, plot=False)
    elif np.sum(np.isnan(df["tissue_2D_KDE%"])):
        for s in df['sample'].unique():
            plot_KDE(df, s, cell_list, plot=False)
    # determine geometric properties of tissue with just this cell type
    tempdf = df.loc[cell_inds,]
    # determine the semi major axis (a), semi minor axis (b), and height (h) of the tissue as an elliptic cylinder
    df['tissue_semimajor'] = np.nan
    df['tissue_semiminor'] = np.nan
    df['tissue_height'] = np.nan
    df['tissue_volume'] = np.nan
    df['tissue_density'] = np.nan
    for s in tempdf['sample'].unique():
        a = np.abs(np.max(tempdf.loc[tempdf["sample"]==s,]["x"]) - np.min(tempdf.loc[tempdf["sample"]==s,]["x"]))/2.0
        b = np.abs(np.max(tempdf.loc[tempdf["sample"]==s,]["y"]) - np.min(tempdf.loc[tempdf["sample"]==s,]["y"]))/2.0
        h = 17 #np.abs(np.max(tempdf.loc[tempdf["sample"]==s]["z"])-np.min(tempdf.loc[tempdf["sample"]==s]["z"])) # TODO: 3D KDE for curved surfaces
        df.loc[df['sample']==s,'tissue_semimajor'] = a
        df.loc[df['sample']==s,'tissue_semiminor'] = b
        df.loc[df['sample']==s,'tissue_height'] = h
        vol = np.pi*a*b*h*df.loc[df['sample']==s, "tissue_2D_KDE%"].unique()[0]
        df.loc[df['sample']==s,'tissue_volume'] = vol
        df.loc[df['sample']==s,'tissue_density'] = np.sum(tempdf["sample"]==s)/vol
    return df

def determine_local_cell_density(df, ref, cell_list, rad=30):
    tempname = " + ".join(cell_list) 
    cell_inds = np.zeros(len(df), dtype='bool')
    for cell in cell_list: cell_inds = cell_inds + np.asarray(df['cell'] == cell)
    if "tissue_density" not in df.columns:
        determine_tissue_geometry(df, cell_list)
    df['local ' + tempname + ' #'] = np.nan  
    df['local ' + tempname + ' density'] = np.nan
    df['local ' + tempname + ' density scaled'] = np.nan
    for i in range(len(df.loc[df['cell']==ref].index)):
        tempind = df.loc[df['cell']==ref].index[i]
        s = df.loc[tempind,'sample']
        temp_cells = df.loc[cell_inds * np.array(df["sample"]==s),][["x", "y", "z"]]
        temp_xyz = np.sqrt(np.sum((df.loc[tempind, ["x", "y", "z"]] - temp_cells)**2, axis = 1))
        h = df.loc[tempind,'tissue_height']
        ln_density = df.loc[tempind,'tissue_density']
        vol = np.pi*rad**2*h
        df.loc[tempind,'local ' + tempname + ' #'] = np.sum(temp_xyz <= rad)
        df.loc[tempind,'local ' + tempname + ' density'] = np.sum(temp_xyz <= rad)/vol
        df.loc[tempind,'local ' + tempname + ' density scaled'] = (np.sum(temp_xyz <= rad)/vol)/ln_density
    return df

def determine_cell_association(df, ref, cell_list, rad=30):
    tempname = ref + " index"
    cell_inds = np.zeros(len(df), dtype='bool')
    for cell in cell_list: cell_inds = cell_inds + np.asarray(df['cell'] == cell)
    df[tempname] = None
    df[tempname]=df[tempname].astype('object')
    for i in range(len(df.loc[df['cell']==ref].index)):
        refind = df.loc[df['cell']==ref].index[i]
        s = df.loc[refind,'sample']
        tempind = df.loc[cell_inds * np.asarray(df['sample']==s)].index
        temp_xyz = np.sqrt(np.sum((df.loc[refind, ["x", "y", "z"]] - df.loc[tempind, ["x", "y", "z"]])**2, axis = 1))
        closeind = np.asarray(temp_xyz<=rad) 
        nanind = np.asarray(df.loc[tempind,tempname].isnull())
        df.loc[tempind[closeind * nanind], tempname] = refind
        listind = tempind[(~nanind)*closeind]
        for j in range(len(listind)):
            if isinstance(df.loc[listind[j], tempname], list):
                tempval = df.loc[listind[j], tempname]
            else:
                tempval = [df.loc[listind[j], tempname]]
            tempval.extend([refind])
            df.at[listind[j], tempname] = tempval
    return df

def null_permutation_MC(df, swap1, swap2, ref_list, cell_list, swap_type=1, rad=30, niter=499):
    ref_inds = np.zeros(len(df), dtype='bool')
    for ref in ref_list: ref_inds = ref_inds + np.asarray(df['cell'] == ref)
    cell_inds = np.zeros(len(df), dtype='bool')
    for cell in cell_list: cell_inds = cell_inds + np.asarray(df['cell'] == cell)
    swap_inds = np.zeros(len(df), dtype='bool')
    for swap in [swap1, swap2]: swap_inds = swap_inds + np.asarray(df['cell'] == swap)
    temp_cells = ref_list; temp_cells.extend(cell_list); temp_cells.append(swap1); temp_cells.append(swap2)
    sample_set = set(df.loc[df['cell']==temp_cells[0]]['sample'].unique())
    for i in range(1, len(temp_cells)):
        sample_set = sample_set & set(df.loc[df['cell']==temp_cells[i]]['sample'].unique())
    sample_inds = np.zeros(len(df), dtype='bool')
    for sample in sample_set: sample_inds = sample_inds + np.asarray(df['sample'] == sample)
    tempdf = df.loc[sample_inds * ref_inds]
    null_shuffles = np.zeros([niter, len(tempdf)])
    sind = 0
    for s in tempdf['sample'].unique():
        h = tempdf.loc[tempdf['sample']==s,'tissue_height'].unique()[0]
        ln_density = tempdf.loc[tempdf['sample']==s,'tissue_density'].unique()[0]
        vol = np.pi*rad**2*h
        for iter in range(niter):
            swap_cells = df.loc[swap_inds * np.array(df["sample"]==s),]
            temp_cells = df.loc[cell_inds * np.array(df["sample"]==s),]
            old_labels = np.asarray(swap_cells['cell'])
            idx = np.random.permutation(len(swap_cells))
            new_labels = old_labels[idx]
            if swap_type: # swapping cell locations 
                temp_r = tempdf
                temp_c = swap_cells.loc[np.array(new_labels==swap1),["x", "y", "z"]]
                temp_inds = tempdf.loc[tempdf['sample']==s].index
            else: # swapping reference cell locations
                temp_r = swap_cells.loc[np.array(new_labels==swap1),["x", "y", "z"]]
                temp_c = temp_cells
                temp_inds = temp_r.index
            # shuffle 
            for i in range(len(temp_inds)):
                temp_xyz = np.sqrt(np.sum((temp_r.loc[temp_inds[i],][["x", "y", "z"]] - temp_c)**2, axis = 1))
                local_density_scaled = (np.sum(temp_xyz<= rad)/vol)/ln_density
                null_shuffles[iter,sind+i] = local_density_scaled
        sind += len(temp_inds)
    return null_shuffles


def null_permutation_pvalue(null_shuffles, actual_value, niter=499):
    r = np.sum(np.mean(null_shuffles, axis=1) >= np.mean(actual_value))
    return (r+1)/(niter+1)

def set_associated_cell_markers(df, ref, cell_list, markers, rad=30):
    tempname = ref + " index"
    cell_inds = np.zeros(len(df), dtype='bool')
    for cell in cell_list: cell_inds = cell_inds + np.asarray(df['cell'] == cell)
    if tempname not in df.columns:
        df = imutil.determine_cell_association(df, ref, cell_list, rad=rad)
    # create columns for associated cell types
    for i in range(len(markers)):
        df[ref +  " " + markers[i]] = np.nan
    # set mean associated cell marker to other cell
    tempnull = ~df[tempname].isnull()
    tempinds = df.index[tempnull]
    for i in range(len(tempinds)):
        for j in range(len(markers)):
            df.iloc[tempinds[i]][ref + " " + markers[j]] = np.mean(df.iloc[df.iloc[tempinds[i]][tempname]][markers[j]])
    return df

