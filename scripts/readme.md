# Robot-Language Project Calvin Datadownload

Script to download the three different splits of Calvin or a small debug dataset:

**1.** [Split D->D](http://calvin.cs.uni-freiburg.de/dataset/task_D_D.zip) (166 GB):
```
cd scripts
sh download_data.sh D
```
**2.** [Split ABC->D](http://calvin.cs.uni-freiburg.de/dataset/task_ABC_D.zip) (517 GB)
```
sh download_data.sh ABC
```
**3.** [Split ABCD->D](http://calvin.cs.uni-freiburg.de/dataset/task_ABCD_D.zip) (656 GB)
```
sh download_data.sh ABCD
```

**4.** [Small debug dataset](http://calvin.cs.uni-freiburg.de/dataset/calvin_debug_dataset.zip) (1.3 GB)
```
sh download_data.sh debug
```

You can verify the integrity of the downloaded zips with the following commands:
```
wget http://calvin.cs.uni-freiburg.de/dataset/sha256sum.txt
sha256sum --check --ignore-missing sha256sum.txt
```
