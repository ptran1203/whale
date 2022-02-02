import os
import shutil
from glob import glob

# for img in glob(f'stamp_comp/Sorted_data1/*/*'):
#     if img.endswith('l.png') or img.endswith('r.png'):
#         os.remove(img)

for di in os.listdir('stamp_comp/Sorted_data1'):
    if di == '.DS_Store':
        os.remove(f'stamp_comp/Sorted_data1/{di}')
    elif not os.listdir(f'stamp_comp/Sorted_data1/{di}'):
        os.rmdir(f'stamp_comp/Sorted_data1/{di}')

data_dir = 'stamp_comp/data_collect_invoice'

for imgr in glob(f'{data_dir}/right/*'):
    imgl = imgr.replace("/right", "/left").replace("r.", "l.")
    assert os.path.exists(imgl), imgl
    label = os.path.basename(imgl).split(".")[0][:-1] + "_invoice"
    print(label)
    dest_dir = f'stamp_comp/Sorted_data1/{label}'
    os.makedirs(dest_dir, exist_ok=True)
    shutil.copy(imgr, f'{dest_dir}/{os.path.basename(imgr)}')
    shutil.copy(imgl, f'{dest_dir}/{os.path.basename(imgl)}')