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
wget --header 'Host: uc7ed7e96efb37137017b25c0130.dl.dropboxusercontent.com' --user-agent 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:66.0) Gecko/20100101 Firefox/66.0' --header 'Accept: text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8' --header 'Accept-Language: en-US,en;q=0.5' --referer 'https://www.dropbox.com/' --header 'Upgrade-Insecure-Requests: 1' 'https://uc7ed7e96efb37137017b25c0130.dl.dropboxusercontent.com/cd/0/get/A8Ck88nZsKAngG7aE__YOPOZqxGeWxS752VVlNBUGymw6-vUNjRUETBP9KOSJa-PPO4MsXWcHceviwGABAivwpSOGfy2ZPIf6vb0cXewtNjvsvlXNDiQTrmtxwibTIIslNc/file#' --output-document 'PointcloudsObject3D.zip'
unzip PointcloudsObject3D.zip && rm PointcloudsObject3D.zip