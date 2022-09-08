### Commands 

* Load the modules: `spectre_load_modules` 

* To peep inside reduction .h5:

```
h5dump -d <data_set>  <file> | less
Ex.) 
h5dump -d Horizon.dat GhKerrSchildReductionData.h5 | less
```

* ls and see what the h5 files contain:

```h5ls <file>```