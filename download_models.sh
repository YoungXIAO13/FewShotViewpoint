#!/usr/bin/env bash

wget --header 'Host: ucd06b217b3c1400fbf0de44f3de.dl.dropboxusercontent.com' --user-agent 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:66.0) Gecko/20100101 Firefox/66.0' --header 'Accept: text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8' --header 'Accept-Language: en-US,en;q=0.5' --referer 'https://www.dropbox.com/' --header 'Upgrade-Insecure-Requests: 1' 'https://ucd06b217b3c1400fbf0de44f3de.dl.dropboxusercontent.com/cd/0/get/A8FKPdw6xtCtRjJhb25jng8HIUmu2IrqLJQnGl5VT186luYszTOyP95N6qohW1xM9r0Bsdg3LevcKWkS9n8SGY-ViLZ1TjPiwxwWOmPJSaKq5LUbV3RXkSBq-mWis5vOs1w/file#' --output-document 'FewShotViewpointBaseModels.zip'

unzip FewShotViewpointBaseModels.zip