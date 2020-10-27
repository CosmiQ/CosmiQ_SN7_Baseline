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
import os

import solaris as sol
from solaris.utils.core import _check_gdf_load
from solaris.raster.image import create_multiband_geotiff 


def map_wrapper(x):
    '''For multi-threading'''
    return x[0](*(x[1:]))
    
    
def multithread_polys(param):
    '''Simple wrapper around mask_to_poly_geojson() for multiprocessing
    # https://solaris.readthedocs.io/en/latest/_modules/solaris/vector/mask.html#mask_to_poly_geojson
    # mask_to_poly_geojson(pred_arr, channel_scaling=None, reference_im=None,
    #                          output_path=None, output_type='geojson', min_area=40,
    #                          bg_threshold=0, do_transform=None, simplify=False,
    #                          tolerance=0.5, **kwargs)
    '''
    
    [pred_image, min_area, output_path_pred, output_type, 
             bg_threshold, simplify] = param
    print("output_pred:", os.path.basename(output_path_pred))
    sol.vector.mask.mask_to_poly_geojson(pred_image, 
                                         min_area=min_area, 
                                         output_path=output_path_pred,
                                         output_type=output_type,
                                         bg_threshold=bg_threshold,
                                         simplify=simplify)
                                         
    
def calculate_iou(pred_poly, test_data_GDF):
    """Get the best intersection over union for a predicted polygon.
    Adapted from: https://github.com/CosmiQ/solaris/blob/master/solaris/eval/iou.py, but
    keeps index of test_data_GDF
    
    Arguments
    ---------
    pred_poly : :py:class:`shapely.Polygon`
        Prediction polygon to test.
    test_data_GDF : :py:class:`geopandas.GeoDataFrame`
        GeoDataFrame of ground truth polygons to test ``pred_poly`` against.
    Returns
    -------
    iou_GDF : :py:class:`geopandas.GeoDataFrame`
        A subset of ``test_data_GDF`` that overlaps ``pred_poly`` with an added
        column ``iou_score`` which indicates the intersection over union value.
    """

    # Fix bowties and self-intersections
    if not pred_poly.is_valid:
        pred_poly = pred_poly.buffer(0.0)

    precise_matches = test_data_GDF[test_data_GDF.intersects(pred_poly)]

    iou_row_list = []
    for idx, row in precise_matches.iterrows():
        # Load ground truth polygon and check exact iou
        test_poly = row.geometry
        # Ignore invalid polygons for now
        if pred_poly.is_valid and test_poly.is_valid:
            intersection = pred_poly.intersection(test_poly).area
            union = pred_poly.union(test_poly).area
            # Calculate iou
            iou_score = intersection / float(union)
            gt_idx = idx
        else:
            iou_score = 0
            gt_idx = -1
        row['iou_score'] = iou_score
        row['gt_idx'] = gt_idx
        iou_row_list.append(row)

    iou_GDF = gpd.GeoDataFrame(iou_row_list)
    return iou_GDF

    
def track_footprint_identifiers(json_dir, out_dir,
                          min_iou=0.25, iou_field='iou_score', id_field='Id',
                          reverse_order=False,     
                          verbose=True, super_verbose=False):
    # 对于一个地点（aoi），这个方法只会调用一次
    # 基本上是一个越来越多的状态，并没有删去iou=0的多边形
    # 但由于生成方式近似于迭代，每一个时间切片存出来的文件只考虑了它之前的情况，只有向前相交则取大，没有向后无交则不要
    # 因此要实现向后无交则drop，需要重写一个drop_lonely_polygons方法。这个方法先全部迭代完成（for j, f in enumerate(json_files):），给每一个多边形id赋予一个intersection_count的字段，然后再调用track_footprint_identifiers的循环部分（for j, f in enumerate(json_files):），在（for pred_idx, pred_row in gdf_now.iterrows():）这个循环的两个（nomatch）的else中，判断intersection_count是否 >= 1，如果真则不运行else中的新赋id的相关代码，而是对gdf_master_Out进行drop操作
    '''
    Track footprint identifiers in the deep time stack.
    We need to track the global gdf instead of just the gdf of t-1.
    '''
        
    os.makedirs(out_dir, exist_ok=True)
    
    # set columns for master gdf
    gdf_master_columns = [id_field, iou_field, 'area', 'geometry', 'intersection_count']

    json_files = sorted([f
                for f in os.listdir(os.path.join(json_dir))
                if f.endswith('.geojson') and os.path.exists(os.path.join(json_dir, f))])
    # start at the end and work backwards?
    if reverse_order:
        json_files = json_files[::-1]

    # check if only partical matching has been done (this will cause errors)
    out_files_tmp = sorted([z for z in os.listdir(out_dir) if z.endswith('.geojson')])
    if len(out_files_tmp) > 0:
        if len(out_files_tmp) != len(json_files):
            raise Exception("\nError in:", out_dir, "with N =", len(out_files_tmp), 
                            "files, need to purge this folder and restart matching!\n")
            return
        elif len(out_files_tmp) == len(json_files):
            print("\nDir:", os.path.basename(out_dir), "N files:", len(json_files), 
                  "directory matching completed, skipping...")
            return
    else:
        print("\nMatching json_dir: ", os.path.basename(json_dir), "N json:", len(json_files))
        
    gdf_dict = {}

    # 首先要得到和最后状态比对的gdf_master_final
    # 第一次循环
    for j, f in enumerate(json_files):
        # 在这里，每一个f就是一个地方不同时间的geojson结果，就是一个时间切片
        json_path = os.path.join(json_dir, f)

        # gdf
        # 每一个时间序列，也就是每一张图片，维护一个gdf
        gdf_now = gpd.read_file(json_path)
        # drop value if it exists
        gdf_now = gdf_now.drop(columns=['value'])
        # get area
        gdf_now['area'] = gdf_now['geometry'].area
        # initialize iou, id
        # 让这一列都等于-1
        gdf_now[iou_field] = -1
        gdf_now[id_field] = -1
        gdf_now['intersection_count'] = 0
        # sort by reverse area
        gdf_now.sort_values(by=['area'], ascending=False, inplace=True)
        gdf_now = gdf_now.reset_index(drop=True)
        # reorder columns (if needed)
        # 要在这里加一个intersection count
        gdf_now = gdf_now[gdf_master_columns]    
        id_set = set([])

        if j == 0:
            # Establish initial footprints at Epoch0
            # set id
            gdf_now[id_field] = gdf_now.index.values
            gdf_now[iou_field] = 0
            gdf_now['intersection_count'] = 0
            n_new = len(gdf_now)
            n_matched = 0
            id_set = set(gdf_now[id_field].values)
            gdf_master_Out = gdf_now.copy(deep=True)
        else:
            gdf_master_Out = gdf_dict['master'].copy(deep=True)
            # OUT：在不同的时间序列之间不断增加新的建筑物，如果没有建筑物和它交叠，那么把它更新到OUT那里
            gdf_master_Edit = gdf_dict['master'].copy(deep=True)
            # Edit：在同一时间序列内，不同建筑物之间。每次时间序列向前时，就会更新为上一个的Out。和每一个建筑物poligon做iou，如果有重叠，就drop掉这个edit，保留自己的形状。
            new_id = np.max(gdf_master_Edit[id_field]) + 1
            idx = 0
            n_new = 0
            n_matched = 0
            for pred_idx, pred_row in gdf_now.iterrows():
                # 每一个pred_row就是一个独立的建筑polygon
                idx += 1
                pred_poly = pred_row.geometry
                # get iou overlap
                iou_GDF = calculate_iou(pred_poly, gdf_master_Edit)
                # iou_GDF = iou.calculate_iou(pred_poly, gdf_master_Edit)
                     
                # Get max iou
                if not iou_GDF.empty:
                    max_iou_row = iou_GDF.loc[iou_GDF['iou_score'].idxmax(axis=0, skipna=True)]
                    # sometimes we are get an erroneous id of 0, caused by nan area,
                    max_area = max_iou_row.geometry.area
                    max_intersect_count = max_iou_row.intersection_count
                    if max_area == 0 or math.isnan(max_area):
                        # print("nan area!", max_iou_row, "returning...")
                        raise Exception("\n Nan area!:", max_iou_row, "returning...")
                        return
                      
                    if max_iou_row['iou_score'] > 0:
                        gdf_now.loc[pred_row.name, 'intersection_count'] = max_intersect_count + 1
                        # drop  matched polygon in ground truth
                        # drop的原因：防止一个建筑同时和两个及以上以前的建筑匹配，因此只要匹配了一个，匹配的target gdf就认为它的某个建筑就被领走了
                        gdf_master_Edit = gdf_master_Edit.drop(max_iou_row.name, axis=0)
                        # 还需要通过更新gdf_master_Out的intersection_count，来把intersection_count更新到gdf_dict['master']中去
                        gdf_master_Out.loc[max_iou_row.name, 'intersection_count'] = max_intersect_count + 1
                        
                    else:
                        # no match, 
                        gdf_master_Out.loc[new_id] = [new_id, 0, pred_poly.area, pred_poly, max_intersect_count + 1]
                        new_id += 1
                        n_new += 1
                       
                else:
                    # no match (same exact code as right above)
                    gdf_master_Out.loc[new_id] = [new_id, 0, pred_poly.area, pred_poly, 0]
                    new_id += 1
                    n_new += 1
                    
        gdf_dict[f] = gdf_now
        # 下面这一行就实现了gdf_dict['master']的不断增量更新
        gdf_dict['master'] = gdf_master_Out

    # 这样就得到了最后状态的gdf
    gdf_master_Final = gdf_dict['master'].copy(deep=True)

    # 第二次循环
    for j, f in enumerate(json_files):
        # 在这里，每一个f就是一个地方不同时间的不同geojson结果
        
        name_root = f.split('.')[0]
        json_path = os.path.join(json_dir, f)
        output_path = os.path.join(out_dir, f)
        
        if verbose and ((j % 1) == 0):
            print("  ", j, "/", len(json_files), "for", os.path.basename(json_dir), "=", name_root)

        # gdf
        # 每一个时间序列，也就是每一张图片，维护一个gdf
        gdf_now = gpd.read_file(json_path)
        # drop value if it exists
        gdf_now = gdf_now.drop(columns=['value'])
        # get area
        gdf_now['area'] = gdf_now['geometry'].area
        # initialize iou, id
        # 让这一列都等于-1
        gdf_now[iou_field] = -1
        gdf_now[id_field] = -1
        # sort by reverse area
        gdf_now.sort_values(by=['area'], ascending=False, inplace=True)
        gdf_now = gdf_now.reset_index(drop=True)
        # reorder columns (if needed)
        # 要在这里加一个intersection count
        gdf_now = gdf_now[gdf_master_columns]    
        id_set = set([])
                           
        if verbose:
            print("\n")
            print("", j, "file_name:", f)
            print("  ", "gdf_now.columns:", gdf_now.columns)
        
        if j == 0:
            # Establish initial footprints at Epoch0
            # set id
            gdf_now[id_field] = gdf_now.index.values
            gdf_now[iou_field] = 0
            n_new = len(gdf_now)
            n_matched = 0
            id_set = set(gdf_now[id_field].values)
            # gdf_dict[f] = gdf_now
        else:
            # match buildings in epochT to epochT-1
            # see: https://github.com/CosmiQ/solaris/blob/master/solaris/eval/base.py
            # print("gdf_master;", gdf_dict['master']) #gdf_master)
            gdf_master_Out = gdf_dict['master'].copy(deep=True)
            # OUT：在不同的时间序列之间不断增加新的建筑物，如果没有建筑物和它交叠，那么把它更新到OUT那里
            gdf_master_Edit = gdf_dict['master'].copy(deep=True)
            # Edit：在同一时间序列内，不同建筑物之间。每次时间序列向前时，就会更新为上一个的Out。和每一个建筑物poligon做iou，如果有重叠，就drop掉这个edit
            if verbose:
                print("   len gdf_now:", len(gdf_now), "len(gdf_master):", len(gdf_master_Out),
                      "max master id:", np.max(gdf_master_Out[id_field]))
                print("   gdf_master_Edit.columns:", gdf_master_Edit.columns)
        
            new_id = np.max(gdf_master_Edit[id_field]) + 1
            idx = 0
            n_new = 0
            n_matched = 0
            gdf_now_copy = gdf_now.copy(deep=True)
            for pred_idx, pred_row in gdf_now_copy.iterrows():
                if verbose:
                    if (idx % 1000) == 0:
                        print("    ", name_root, idx, "/", len(gdf_now))
                if super_verbose:
                    # print("    ", i, j, idx, "/", len(gdf_now))
                    print("    ", idx, "/", len(gdf_now))
                idx += 1
                pred_poly = pred_row.geometry
                    
                # get iou overlap
                iou_GDF = calculate_iou(pred_poly, gdf_master_Edit)
                # iou_GDF = iou.calculate_iou(pred_poly, gdf_master_Edit)
                     
                # Get max iou
                if not iou_GDF.empty:
                    max_iou_row = iou_GDF.loc[iou_GDF['iou_score'].idxmax(axis=0, skipna=True)]
                    # sometimes we are get an erroneous id of 0, caused by nan area,
                    max_area = max_iou_row.geometry.area
                    if max_area == 0 or math.isnan(max_area):
                        raise Exception("\n Nan area!:", max_iou_row, "returning...")
                        return
                    
                    id_match = max_iou_row[id_field]
                    if id_match in id_set:
                        print("Already seen id! returning...")
                        raise Exception("\n Already seen id!", id_match, "returning...")
                        return
                      
                    if max_iou_row['iou_score'] >= min_iou:
                        if super_verbose:
                            print("    pred_idx:", pred_idx, "match_id:", max_iou_row[id_field],
                                  "max iou:", max_iou_row['iou_score'])
                        gdf_now.loc[pred_row.name, iou_field] = max_iou_row['iou_score']
                        gdf_now.loc[pred_row.name, id_field] = id_match
                        # 不要intersection_count=0的gdf行
                        if max_iou_row['intersection_count'] = 0:
                            gdf_now.drop(pred_row.name, axis=0)
                        gdf_master_Edit = gdf_master_Edit.drop(max_iou_row.name, axis=0)
                        n_matched += 1
                      
                    else:
                        # no match, 
                        if super_verbose:
                            print("    Minimal match! - pred_idx:", pred_idx, "match_id:",
                                  max_iou_row[id_field], "max iou:", max_iou_row['iou_score'])
                            print("      Using new id:", new_id)
                        if (new_id in id_set) or (new_id == 0):
                            raise Exception("trying to add an id that already exists, returning!")
                            return
                        gdf_now.loc[pred_row.name, iou_field] = 0
                        gdf_now.loc[pred_row.name, id_field] = new_id
                        id_set.add(new_id)
                        # update master, cols = [id_field, iou_field, 'area', 'geometry']
                        gdf_master_Out.loc[new_id] = [new_id, 0, pred_poly.area, pred_poly, 0]
                        new_id += 1
                        n_new += 1
                       
                else:
                    # no match (same exact code as right above)
                    if super_verbose:
                        print("    pred_idx:", pred_idx, "no overlap, new_id:", new_id)
                    if (new_id in id_set) or (new_id == 0):
                        raise Exception("trying to add an id that already exists, returning!")
                        return
                    gdf_now.loc[pred_row.name, iou_field] = 0
                    gdf_now.loc[pred_row.name, id_field] = new_id
                    id_set.add(new_id)
                    # update master, cols = [id_field, iou_field, 'area', 'geometry']
                    gdf_master_Out.loc[new_id] = [new_id, 0, pred_poly.area, pred_poly, 0]
                    new_id += 1
                    n_new += 1
                    
        # print("gdf_now:", gdf_now)
        gdf_dict[f] = gdf_now

        # save!
        if len(gdf_now) > 0:
            gdf_now.to_file(output_path, driver="GeoJSON")
            # 每一次循环都会存一次，假设20个时间序列，则会同样存出20个经过iou比对的东西
        else:
            print("Empty dataframe, writing empty gdf", output_path)
            open(output_path, 'a').close()

        if verbose:
            print("  ", "N_new, N_matched:", n_new, n_matched)
         
    return 
  

def sn7_convert_geojsons_to_csv(json_dirs, output_csv_path, population='proposal'):
    '''
    Convert jsons to csv
    Population is either "ground" or "proposal" 
    '''
    
    first_file = True  # switch that will be turned off once we process the first file
    for json_dir in tqdm.tqdm(json_dirs):
        json_files = sorted(glob.glob(os.path.join(json_dir, '*.geojson')))
        for json_file in tqdm.tqdm(json_files):
            try:
                df = gpd.read_file(json_file)
            except (fiona.errors.DriverError):
                message = '! Invalid dataframe for %s' % json_file
                print(message)
                continue
                #raise Exception(message)
            if population == 'ground':
                file_name_col = df.image_fname.apply(lambda x: os.path.splitext(x)[0])
            elif population == 'proposal':
                file_name_col = os.path.splitext(os.path.basename(json_file))[0]
            else:
                raise Exception('! Invalid population')
            df = gpd.GeoDataFrame({
                'filename': file_name_col,
                'id': df.Id.astype(int),
                'geometry': df.geometry,
            })
            if len(df) == 0:
                message = '! Empty dataframe for %s' % json_file
                print(message)
                #raise Exception(message)

            if first_file:
                net_df = df
                first_file = False
            else:
                net_df = net_df.append(df)
                
    net_df.to_csv(output_csv_path, index=False)
    return net_df