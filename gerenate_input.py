import numpy as np
import csv
from concatenate_features import create_clip_extract_features
"""
    generate .csv files
    each file contains one maneuver data, 12 features (means and variances)
"""

# generate lane keeping features
lk_feat = create_clip_extract_features('lk')
# write in .csv file
with open('lk_rvm_feat.csv', 'w') as writeFile:
    writer = csv.writer(writeFile)
    writer.writerows(lk_feat)
writeFile.close()

# generate right lane change features
rightlc_feat = create_clip_extract_features('rightlc')
# write in .csv file
with open('rightlc_rvm_feat.csv', 'w') as writeFile:
    writer = csv.writer(writeFile)
    writer.writerows(rightlc_feat)
writeFile.close()

# generate left lane change features
leftlc_feat = create_clip_extract_features('leftlc')
# write in .csv file
with open('leftlc_rvm_feat.csv', 'w') as writeFile:
    writer = csv.writer(writeFile)
    writer.writerows(leftlc_feat)
writeFile.close()
