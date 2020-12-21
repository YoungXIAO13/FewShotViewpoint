## Download dataset from official website
wget 'ftp://cs.stanford.edu/cs/cvgl/PASCAL3D+_release1.1.zip'
unzip PASCAL3D+_release1.1.zip && rm PASCAL3D+_release1.1.zip
mv PASCAL3D+_release1.1/* ./ && rm -r PASCAL3D+_release1.1

## Create annotation file
python create_annotation.py

## Download the point clouds for this dataset
wget "https://www.dropbox.com/s/7bpooss6iqff7kc/Pascal3DPointclouds.zip?dl=0" && mv Pascal3DPointclouds.zip?dl=0 Pascal3DPointclouds.zip
unzip Pascal3DPointclouds.zip && rm Pascal3DPointclouds.zip
