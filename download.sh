#!/bin/bash
# --------------------------------------------------------------
# This script download a CICIDS2017 MachineLearningCSV dataset.
#
# Author: Mahendra Data - mahendra.data@dbms.cs.kumamoto-u.ac.jp
# License: BSD 3 clause
#
# Running example: "bash download.sh"
# --------------------------------------------------------------

NAME="CICIDS2017-MachineLearning"

echo "Downloading CICIDS2017. If the dataset already downloaded, then skip it."
wget -nc -O $NAME.zip \
  http://205.174.165.80/CICDataset/CIC-IDS-2017/Dataset/MachineLearningCSV.zip

echo "Downloading checksum."
wget -nc -O $NAME.md5 \
  http://205.174.165.80/CICDataset/CIC-IDS-2017/Dataset/MachineLearningCSV.md5

echo "Replacing 'MachineLearningCVE.zip' in ${NAME}.md5 with '${NAME}.zip'."
sed -i "s/MachineLearningCVE.zip/${NAME}.zip/g" ${NAME}.md5

echo "Checking dataset file integrity."
md5sum --check ${NAME}.md5

echo "Unzip-ing the dataset."
unzip -n $NAME".zip"

echo "Renaming the extracted dataset directory."
mv "MachineLearningCVE/" $NAME