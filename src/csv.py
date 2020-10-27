import os
from os.path import dirname, join
from sys import argv

pop = argv[1]

src = dirname(__file__)
code = dirname(src)
tmp = join(code, 'tmp/')
data = join(code, f'{pop}/')
outpath = join(tmp, f'{pop}.csv')

im_list, mask_list = [], []
subdirs = sorted([f for f in os.listdir(data) if os.path.isdir(os.path.join(data, f))])
for subdir in subdirs:
    if pop == 'train':
        im_files = [os.path.join(data, subdir, 'images_masked', f)
                for f in sorted(os.listdir(os.path.join(data, subdir, 'images_masked')))
                if f.endswith('.tif') and os.path.exists(os.path.join(data, subdir, 'masks', f.split('.')[0] + '_Buildings.tif'))]
        mask_files = [os.path.join(d, subdir, 'masks', f.split('.')[0] + '_Buildings.tif')
                    for f in sorted(os.listdir(os.path.join(data, subdir, 'images_masked')))
                    if f.endswith('.tif') and os.path.exists(os.path.join(data, subdir, 'masks', f.split('.')[0] + '_Buildings.tif'))]
        im_list.extend(im_files)
        mask_list.extend(mask_files)
    elif pop == 'test_public':
        im_files = [os.path.join(data, subdir, 'images_masked', f)
                for f in sorted(os.listdir(os.path.join(data, subdir, 'images_masked')))
                if f.endswith('.tif')]
        im_list.extend(im_files)

if pop == 'train':
    df = pd.DataFrame({'image': im_list, 'label': mask_list})
elif pop == 'test_public':
    df = pd.DataFrame({'image': im_list})
df.to_csv(outpath, index=False)
print(pop, "len df:", len(df))
print("output csv:", outpath)
