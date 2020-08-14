#!/usr/bin/env bash

wget --header 'Host: uceeedb649b6a047ef19e91dfdc3.dl.dropboxusercontent.com' --user-agent 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:66.0) Gecko/20100101 Firefox/66.0' --header 'Accept: text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8' --header 'Accept-Language: en-US,en;q=0.5' --referer 'https://www.dropbox.com/' --header 'Upgrade-Insecure-Requests: 1' 'https://uceeedb649b6a047ef19e91dfdc3.dl.dropboxusercontent.com/cd/0/get/A9aodGEIo8O8Z4a9tSNdQujjLJj7Hkb_f1wRJ9zF2ISHA2S9KMggGjpt_zefxggVrl8UOCB-17CKLOocdab1JmAOG3cXGBaEiuaEMe3MDKFdlMgbArbt3GBVkEODmL6y76g/file#' --output-document 'FewShotViewpointBaseModels.zip'

unzip FewShotViewpointBaseModels.zip
