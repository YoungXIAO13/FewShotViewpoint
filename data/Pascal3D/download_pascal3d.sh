## Download dataset from official website
wget 'ftp://cs.stanford.edu/cs/cvgl/PASCAL3D+_release1.1.zip'
unzip PASCAL3D+_release1.1.zip && rm PASCAL3D+_release1.1.zip
mv PASCAL3D+_release1.1/* ./ && rm -r PASCAL3D+_release1.1

## Create annotation file
python create_annotation.py

## Download the point clouds for this dataset
wget --header 'Host: uc394c84ad6d200269554224e5fd.dl.dropboxusercontent.com' --user-agent 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:66.0) Gecko/20100101 Firefox/66.0' --header 'Accept: text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8' --header 'Accept-Language: en-US,en;q=0.5' --referer 'https://www.dropbox.com/' --header 'Upgrade-Insecure-Requests: 1' 'https://uc394c84ad6d200269554224e5fd.dl.dropboxusercontent.com/cd/0/get/A8GUzMttGrrgQBivwmLs4WrelX4KJ12JuBt7xKlP5-KfKxtrEziPmahJZCYguv0zeaVHUzv1_48L8IHEUXZ-GP4AD8cLtomyUA0Ngz1131Oh9-shucEwEa4-hgl-cTh5ivs/file#' --output-document 'Pascal3DPointclouds.zip'
unzip Pascal3DPointclouds.zip && rm Pascal3DPointclouds.zip