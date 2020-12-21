## Download dataset from official website
wget 'ftp://cs.stanford.edu/cs/cvgl/ObjectNet3D/ObjectNet3D_images.zip'
unzip ObjectNet3D_images.zip && rm ObjectNet3D_images.zip
mv ObjectNet3D/* ./ && rm -r ObjectNet3D

wget 'ftp://cs.stanford.edu/cs/cvgl/ObjectNet3D/ObjectNet3D_cads.zip'
unzip ObjectNet3D_cads.zip && rm ObjectNet3D_cads.zip
mv ObjectNet3D/* ./ && rm -r ObjectNet3D

wget 'ftp://cs.stanford.edu/cs/cvgl/ObjectNet3D/ObjectNet3D_annotations.zip'
unzip ObjectNet3D_annotations.zip && rm ObjectNet3D_annotations.zip
mv ObjectNet3D/* ./ && rm -r ObjectNet3D

wget 'ftp://cs.stanford.edu/cs/cvgl/ObjectNet3D/ObjectNet3D_image_sets.zip'
unzip ObjectNet3D_image_sets.zip && rm ObjectNet3D_image_sets.zip
mv ObjectNet3D/* ./ && rm -r ObjectNet3D

## Create annotation file
python create_annotation.py

## Download the point clouds for this dataset
wget "https://www.dropbox.com/s/0s61m3cvwir0tsc/ObjectNet3DPointclouds.zip?dl=0" && mv ObjectNet3DPointclouds.zip?dl=0 ObjectNet3DPointclouds.zip
unzip ObjectNet3DPointclouds.zip && rm ObjectNet3DPointclouds.zip
