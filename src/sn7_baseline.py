# # Train and Test the SpaceNet 7 Baseline Algorithm
# 
# 
# We assume that initial steps of README have been executed, sn7_data_prep.ipynb has been executed, and that this notebook is running in a docker container.  See the `src` directory for functions used in the algorithm.  

# --------
# ## 1. Train
# 1. Launch a separate docker container for training:
# 
#        nvidia-docker build -t sn7_baseline_image /path_to_baseline/docker 
#        NV_GPU=0 nvidia-docker run -it -v /local_data:/local_data  -ti --ipc=host --name sn7_gpu0 sn7_baseline_image
#        conda activate solaris
#     
# 2. Initiate training.  First edit `sn7_baseline_train.yml` to point to the correct data paths, then execute the following in the command line: 
#     
#        cd /path_to_baseline/src
#        time python sn7_baseline_train.py
# 
#     Training for the full 300 epochs should take 20 hours (~$60) on a p3.2xlarge AWS instance.
#     
#  
#  
# 3. Alternately, instead of training, one could use the pre-trained weights included in this repository.

# --------
# ## 2. Infer
# Once training is completed (or pre-trained weights are selected) it's time to initiate inference. First edit `sn7_baseline_infer.yml` to point to the correct data paths, run:
# 
#         cd /path_to_baseline/src
#         time python sn7_baseline_infer.py
# 
# 
# This script will execute in ~2.5 minutes on a p3.2xlarge AWS instance (which equates to  approximately 60 square kilometers per second).

# --------
# ## 3. Extract Footprints and Building Identifiers
# 
# The `sn7_baseline_infer.py` script executes the segmentation model, which is only the first step in the extracting matched building footprints in the data cube.  In the cells below, we refine these predictioms masks to the final output.

# %%
# Set prediction and image directories (edit appropriately)
pred_top_dir = '/workspace/codes/CosmiQ_SN7_Baseline/inference_out/sn7_baseline_preds'
im_top_dir = '/workspace/data/test_public'

from shapely.ops import cascaded_union
import matplotlib.pyplot as plt
import geopandas as gpd
import multiprocessing
import pandas as pd
import numpy as np
import skimage.io
import tqdm
import glob
import math
import gdal
import time
import sys
import os

import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 16})
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300
import matplotlib
# matplotlib.use('Agg') # non-interactive

import solaris as sol
from solaris.utils.core import _check_gdf_load
from solaris.raster.image import create_multiband_geotiff 

# import from data_postproc_funcs
module_path = os.path.abspath(os.path.join('../src/'))
if module_path not in sys.path:
    sys.path.append(module_path)
from sn7_baseline_postproc_funcs import map_wrapper, multithread_polys,         calculate_iou, track_footprint_identifiers,         sn7_convert_geojsons_to_csv


# --------
# ### 3.A. Group predictions by AOI

# %%
raw_name = 'raw/'
grouped_name = 'grouped/'
im_list = sorted([z for z in os.listdir(os.path.join(pred_top_dir, raw_name)) if z.endswith('.tif')])
df = pd.DataFrame({'image': im_list})
roots = [z.split('mosaic_')[-1].split('.tif')[0] for z in df['image'].values]
df['root'] = roots
# copy files
for idx, row in df.iterrows():
    in_path_tmp = os.path.join(pred_top_dir, raw_name, row['image'])
    out_dir_tmp = os.path.join(pred_top_dir, grouped_name, row['root'], 'masks')
    os.makedirs(out_dir_tmp, exist_ok=True)
    cmd = 'cp ' + in_path_tmp + ' ' + out_dir_tmp
    print("cmd:", cmd)
    os.system(cmd)    


# # --------
# # ## 3.B. (Optional) Explore predictions

# # %%
# # Inspect visually

# aoi = 'L15-0509E-1108N_2037_3758_13'
# out_dir_explore = os.path.join(pred_top_dir, 'explore', aoi)
# os.makedirs(out_dir_explore, exist_ok=True)

# pred_dir = os.path.join(pred_top_dir, 'grouped', aoi, 'masks')
# im_dir = os.path.join(im_top_dir, aoi, 'images_masked')
# im_list = sorted([z for z in os.listdir(pred_dir) if z.endswith('.tif')])
# sample_mask_name = im_list[0]
# sample_mask_path = os.path.join(pred_dir, sample_mask_name)
# sample_im_path = os.path.join(im_dir, sample_mask_name)

# image = skimage.io.imread(sample_im_path)
# mask_image = skimage.io.imread(sample_mask_path)
# print("mask_image.shape:", mask_image.shape)
# print("min, max, mean mask image:", np.min(mask_image), np.max(mask_image), np.mean(mask_image))

# # # vertical layout
# # figsize = (14, 14)
# # fig, ax = plt.subplots(figsize=figsize)
# # _ = ax.imshow(image)
# # ax.set_title(sample_mask_name)
# # plt.savefig(os.path.join(out_dir_explore, aoi + '_im0.png'))
# # fig, ax = plt.subplots(figsize=figsize)
# # _ = ax.imshow(mask_image)
# # ax.set_title(sample_mask_name)
# # plt.savefig(os.path.join(out_dir_explore, aoi + '_mask0.png'))
# # plt.show()

# # horizontal 
# figsize = (20, 10)
# fig, (ax0, ax1) = plt.subplots(1, 2, figsize=figsize)
# _ = ax0.imshow(image)
# ax0.set_xticks([])
# ax0.set_yticks([])
# ax0.set_title('Image')
# _ = ax1.imshow(mask_image)
# ax1.set_xticks([])
# ax1.set_yticks([])
# ax1.set_title('Prediction Mask')
# plt.suptitle(sample_mask_name.split('.')[0])
# plt.tight_layout()
# plt.savefig(os.path.join(out_dir_explore, aoi + '_im0+mask0.png'))
# plt.show()

# # zoom in a subset 
# # horizontal 
# figsize = (20, 10)
# bounds = []
# fig, (ax0, ax1) = plt.subplots(1, 2, figsize=figsize)
# _ = ax0.imshow(image[400:600,400:600,:])
# ax0.set_xticks([])
# ax0.set_yticks([])
# ax0.set_title('Image')
# _ = ax1.imshow(mask_image[400:600,400:600])
# ax1.set_xticks([])
# ax1.set_yticks([])
# ax1.set_title('Prediction Mask')
# plt.suptitle(sample_mask_name.split('.')[0])
# plt.tight_layout()
# plt.savefig(os.path.join(out_dir_explore, aoi + '_im0+mask0_zoom.png'))
# plt.show()


# # %%
# # Extract and inspect sample footprints
# # https://solaris.readthedocs.io/en/latest/tutorials/notebooks/api_mask_to_vector.html
# # https://github.com/CosmiQ/solaris/blob/master/solaris/vector/mask.py#L718

# output_path_pred = os.path.join(out_dir_explore, aoi + '_pred0.geojson')
# min_area = 3.5    # in pixels
# bg_threshold = 0
# simplify = False
# print("bg_threshold:", bg_threshold)
# print("min_area:", min_area)
# geoms = sol.vector.mask.mask_to_poly_geojson(mask_image, 
#                                              min_area=min_area, 
#                                              output_path=output_path_pred,
#                                              output_type='geojson',
#                                              bg_threshold=bg_threshold,
#                                              simplify=simplify)
# display(geoms.head())
# print("N geoms:", len(geoms))

# # get plot geoms
# plot_geoms = cascaded_union(geoms['geometry'])
# # display(plot_geoms)


# # %%
# # Plot each polygon shape
# fig, ax = plt.subplots(figsize=(14, 14))
# for geom in plot_geoms.geoms:
#     ax.plot(*geom.exterior.xy)
# ax.set_xlim(0, mask_image.shape[1])
# ax.set_ylim(mask_image.shape[1], 0)
# ax.set_aspect('equal')
# # Set (current) axis to be equal before showing plot?
# # plt.gca().axis("equal")
# plt.savefig(os.path.join(out_dir_explore, aoi + '_footprints0.png'))
# plt.show()


# ------
# ## 3.C. Extract building footprint geometries for all AOIs

# %%
# Get all geoms for all aois (mult-threaded)

min_area = 3.5   # in pixels (4 is standard)
simplify = False
bg_threshold = 0  
output_type = 'geojson'
aois = sorted([f for f in os.listdir(os.path.join(pred_top_dir, 'grouped')) if os.path.isdir(os.path.join(pred_top_dir, 'grouped', f))])

# set params
params = []
for i, aoi in enumerate(aois):
    print(i, "/", len(aois), aoi)   
    outdir = os.path.join(pred_top_dir, 'grouped', aoi, 'pred_jsons')
    os.makedirs(outdir, exist_ok=True)
    pred_files = sorted([os.path.join(pred_top_dir, 'grouped', aoi, 'masks', f)
                for f in sorted(os.listdir(os.path.join(pred_top_dir, 'grouped', aoi, 'masks')))
                if f.endswith('.tif')])
    for j, p in enumerate(pred_files):
        name = os.path.basename(p)
        # print(i, j, name)
        output_path_pred = os.path.join(outdir,  name.split('.tif')[0] + '.geojson')
        # get pred geoms
        if not os.path.exists(output_path_pred):
            pred_image = skimage.io.imread(p)#[:,:,0]
            params.append([pred_image, min_area, output_path_pred,
                          output_type, bg_threshold, simplify])        

print("Execute!")
print("len params:", len(params))
n_threads = 10
pool = multiprocessing.Pool(n_threads)
_ = pool.map(multithread_polys, params)


# ----------
# ## 3.D. Track building identifiers
# 
# Now we assign a unique identifier to each building, and propogate that identifier through the data cube.

# %%
# This takes awhile, so multi-thread it

min_iou = 0.2
iou_field = 'iou_score'
id_field = 'Id'
reverse_order = False
verbose = True
super_verbose = False
n_threads = 10

json_dir_name = 'pred_jsons/'
out_dir_name = 'pred_jsons_match/'
aois = sorted([f for f in os.listdir(os.path.join(pred_top_dir, 'grouped')) 
               if os.path.isdir(os.path.join(pred_top_dir, 'grouped', f))])
print("aois:", aois)

print("Gather data for matching...")
params = []
for aoi in aois:
    print(aoi)
    json_dir = os.path.join(pred_top_dir, 'grouped', aoi, json_dir_name)
    out_dir = os.path.join(pred_top_dir, 'grouped', aoi, out_dir_name)
    
    # check if we started matching...
    if os.path.exists(out_dir):
        # print("  outdir exists:", outdir)
        json_files = sorted([f
                for f in os.listdir(os.path.join(json_dir))
                if f.endswith('.geojson') and os.path.exists(os.path.join(json_dir, f))])
        out_files_tmp = sorted([z for z in os.listdir(out_dir) if z.endswith('.geojson')])
        if len(out_files_tmp) > 0:
            if len(out_files_tmp) == len(json_files):
                print("Dir:", os.path.basename(out_dir), "N files:", len(json_files), 
                      "directory matching completed, skipping...")
                continue
            elif len(out_files_tmp) != len(json_files):
                # raise Exception("Incomplete matching in:", out_dir, "with N =", len(out_files_tmp), 
                #                 "files (should have N_gt =", 
                #                 len(json_files), "), need to purge this folder and restart matching!")
                print("Incomplete matching in:", out_dir, "with N =", len(out_files_tmp), 
                                "files (should have N_gt =", 
                                len(json_files), "), purging this folder and restarting matching!")
                purge_cmd = 'rm -r ' + out_dir
                print("  purge_cmd:", purge_cmd)
                if len(out_dir) > 20:
                    purge_cmd = 'rm -r ' + out_dir
                else:
                    raise Exception("out_dir too short, maybe deleting something unintentionally...")
                    break
                os.system(purge_cmd)
            else:
                pass

    params.append([track_footprint_identifiers, json_dir,  out_dir, min_iou, 
                   iou_field, id_field, reverse_order, verbose, super_verbose])    

print("Len params:", len(params))


# %%
print("Execute!")
n_threads = 10
pool = multiprocessing.Pool(n_threads)
_ = pool.map(map_wrapper, params)


# # %%
# # make plots (optional)

# # %matplotlib notebook
# im_pix_size_x, im_pix_size_y = 1024, 1024
# max_plots = 2
# label_font_size = 5
# figsize = (16, 16)

# aois = ['L15-1281E-1035N_5125_4049_13']
# print("aois:", aois)

# count = 0
# for i, aoi in enumerate(aois):
#     print("\n")
#     print(i, "aoi:", aoi)
    
#     json_files = sorted([f
#                 for f in os.listdir(os.path.join(pred_top_dir, 'grouped', aoi, 'pred_jsons_match'))
#                 if f.endswith('.geojson') and os.path.exists(os.path.join(pred_top_dir, 'grouped', aoi, 'pred_jsons_match', f))])
#     # take only the first and last?
#     # json_files = [json_files[0], json_files[-1]]
#     # plot 
#     for j, f in enumerate(json_files):
#         if count >= max_plots:
#             break
#         else:
#             count += 1
#         print(i, j, f)
#         name_root = f.split('.')[0]
#         json_path = os.path.join(pred_top_dir, 'grouped', aoi, 'pred_jsons_match', f)
#         print("name_root:", name_root)
#         # print("json_path:", json_path)
#         gdf_pix = _check_gdf_load(json_path)
#         fig, ax = plt.subplots(figsize=figsize)
#         for _, row in gdf_pix.iterrows():
#             geom = row['geometry']
#             poly_id = row['Id']
#             x, y = geom.exterior.xy
#             cx, cy = np.array(geom.centroid.xy).astype(float)
#             # print("centroid:", centroid)
#             ax.plot(x, y)
#             # poly id
#             ax.annotate(str(poly_id), xy=(cx, cy), ha='center', size=label_font_size)
#             # text_object = plt.annotate(label, xy=(x_values[i], y_values[i]), ha='center')
#             # ax.text(cx, cy, str(poly_id))
#         ax.set_xlim(0, im_pix_size_x)
#         ax.set_ylim(0, im_pix_size_y)
#         title = str(j) + " - " + name_root + " - N buildings = " + str(len(gdf_pix))
#         ax.set_title(title)
#         plt.tight_layout()
        
#     plt.show()
        


# --------
# ## 3.E. Make proposal CSV 
# This is necessary for scoring with the [SCOT metric](https://github.com/CosmiQ/solaris/blob/master/solaris/eval/scot.py).

# %%
# Make proposal csv

out_dir_csv = os.path.join(pred_top_dir, 'csvs')
os.makedirs(out_dir_csv, exist_ok=True)
prop_file = os.path.join(out_dir_csv, 'sn7_baseline_predictions.csv')

aoi_dirs = sorted([os.path.join(pred_top_dir, 'grouped', aoi, 'pred_jsons_match')                    for aoi in os.listdir(os.path.join(pred_top_dir, 'grouped'))                    if os.path.isdir(os.path.join(pred_top_dir, 'grouped', aoi, 'pred_jsons_match'))])
print("aoi_dirs:", aoi_dirs)

# Execute
if not os.path.exists(prop_file):
    net_df = sn7_convert_geojsons_to_csv(aoi_dirs, prop_file, 'proposal')

print("prop_file:", prop_file)


# --------
# 
# ## 4. Conclusions
# 
# The notebook above walks through all the steps to train and test a model that extracts building footprints with (hopefully) persistent unique identifiers (i.e. addresses) from a deep temporal stack of medium resolution satellite imagery.  
# 
# The model proposed here achieves a [SCOT](https://github.com/CosmiQ/solaris/blob/master/solaris/eval/scot.py) score of 0.158 on the SpaceNet 7 test_public data.
# 

# %%


