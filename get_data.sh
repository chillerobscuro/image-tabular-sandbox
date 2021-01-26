#!/bin/bash

mkdir data/
curl 'https://dokruq.db.files.1drv.com/y4mOpkZMYeo1X01emWotOFou8x5bwh0OliMhjbJSgu3Z3zLX7OuTQ4tvDF7xl_SONH7vX506GVATBf6AZardRf7rcWBgPiA8XNT6dWHvsFUeaiJ5UAzVqrqXaXXohJkTYSYKv0daHM3iRFUz5uf6tY1xvipUuzzWmJiBe-EPgwTfftn8gu2SovPwiZ8XthhQxTt0Yapzr_-itMcdyH1YeogxA' -H 'User-Agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10.14; rv:84.0) Gecko/20100101 Firefox/84.0' -H 'Accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8' -H 'Accept-Language: en-US,en;q=0.5' --compressed -H 'Referer: https://onedrive.live.com/' -H 'Connection: keep-alive' -H 'Upgrade-Insecure-Requests: 1' -H 'TE: Trailers' --output data/data.zip
cd data/ || exit
# if following command doesn't work, run: sudo apt-get install unzip
unzip data.zip
