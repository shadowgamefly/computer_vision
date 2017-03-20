## Stereo Matching

To run both program simply do the following:
```bash
$ python3 panaroma.py
```
``` bash
$ python3 stereo.py
```
### Result

* Gaussian Filter
  * RMS = 12.743754113606503
  * sigma = 0.9
  * Image see `Gaussian.png`

* Bilateral Filter
  * RMS = 12.74
  * d = 3, sigmaColor = 3, sigmaSpace = 0.5
  * Image see  `Bilateral.png`

* Left to Right Check
  * RMS = 9.362459994733678
  * Image see `leftToRight.png`

## Panorama stitching

* best H:

 [   0.61684454   -0.21092917  372.40441395]

 [  -0.07643232    0.64380831   81.07477423]

 [  -0.00037545   -0.00048571   1        ]
