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
wget --header 'Host: uc23d47b3615cc6557d59b6a3ae5.dl.dropboxusercontent.com' --user-agent 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:66.0) Gecko/20100101 Firefox/66.0' --header 'Accept: text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8' --header 'Accept-Language: en-US,en;q=0.5' --referer 'https://www.dropbox.com/' --header 'Upgrade-Insecure-Requests: 1' 'https://uc23d47b3615cc6557d59b6a3ae5.dl.dropboxusercontent.com/cd/0/get/A8GKqmoG-OuJ_z0eMpz12RGIfC8UmGoK1_gkgL3tskQz4WydJr50I_SqcdywnzBoca01TpfUAf6rCXlifu6ss6MqKD2XjA1ooD90Ge9qeQNtTo1Nl-q7kXWrCHkKy0lenLM/file#' --output-document 'ObjectNet3DPointclouds.zip'
unzip ObjectNet3DPointclouds.zip && rm ObjectNet3DPointclouds.zip