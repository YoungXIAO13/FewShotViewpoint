#!/usr/bin/env bash

# Download Base Models obtained after the first stage base class training
wget https://www.dropbox.com/s/7syiwo2guy06xhr/FewShotViewpointBaseModels.zip?dl=0 && mv FewShotViewpointBaseModels.zip?dl=0 FewShotViewpointBaseModels.zip
unzip FewShotViewpointBaseModels.zip && rm FewShotViewpointBaseModels.zip


# Download Trained Models obtained after the second stage few-shot fine-tuning
wget https://www.dropbox.com/s/c86c32kmr040ryo/InterDataset_shot10.zip?dl=0 && mv InterDataset_shot10.zip?dl=0 InterDataset_shot10.zip
unzip InterDataset_shot10.zip && rm InterDataset_shot10.zip

wget https://www.dropbox.com/s/6r3hltomzkehbfp/IntraDataset_shot10.zip?dl=0 && mv IntraDataset_shot10.zip?dl=0 IntraDataset_shot10.zip
unzip IntraDataset_shot10.zip && rm IntraDataset_shot10.zip && rm -r __MACOSX/
