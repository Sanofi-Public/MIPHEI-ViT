Clone HoverFast repository: https://github.com/choosehappy/HoverFast


Code adaptation: # replace openslide by slidevips

- replace main.py here ->  hoverfast/main.py
- replace utils_wsi.py -> hoverfast/utils_wsi.py

Then run inference like this:

HoverFast infer_wsi /data/*.ome.tiff -m ../HoverFast/hoverfast_crosstissue_best_model.pth -l 20 -s 2 -o /data/hoverfast_output -r 100
