## Download dataset from official website
wget 'ftp://cs.stanford.edu/cs/cvgl/PASCAL3D+_release1.1.zip'
unzip PASCAL3D+_release1.1.zip && rm PASCAL3D+_release1.1.zip
mv PASCAL3D+_release1.1/* ./ && rm -r PASCAL3D+_release1.1

## Create annotation file
python create_annotation.py

## Download the point clouds for this dataset
wget --header 'Host: ucd137deee69783344e7f84e6783.dl.dropboxusercontent.com' --user-agent 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:66.0) Gecko/20100101 Firefox/66.0' --header 'Accept: text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8' --header 'Accept-Language: en-US,en;q=0.5' --referer 'https://www.dropbox.com/' --header 'Upgrade-Insecure-Requests: 1' 'https://ucd137deee69783344e7f84e6783.dl.dropboxusercontent.com/cd/0/get/A8B3IvwJ_YGt2ZOy0dXvBmTxJ5f5JvXP7sJOV4XxnpMrn3YDwSVH73NppnHXE5jJXcvWjIw4g6mGWa-1oxJjdbHyLZxmCqBDDCM4M09V-r3S0IFTTuzZoMQ80f94n2H-Q18/file#' --output-document 'PointcloudsPascal3D.zip'
unzip PointcloudsPascal3D.zip && rm PointcloudsPascal3D.zip