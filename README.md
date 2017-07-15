# CS Relief Map Tile Image Generator

## What is CS Relief Map (CS立体図) ?

by Nagano prefecture forest center [長野県林業総合センター](https://www.pref.nagano.lg.jp/ringyosogo/)

this implementation is based on these articles

 - [数値地形データを用いた「微地形図」の作成方法](http://www.pref.nagano.lg.jp/ringyosogo/seika/documents/bichikei.pdf)
 - [QGISで「CS立体図」の「立体図」を作ってみた](http://koutochas.seesaa.net/article/444171690.html)


### Setup

```python
sudo pip install -r requirements.txt
```

### Examples

#### generate by coordinate

```python
# North of Hokkaido
python CSReliefMap.py generateImageByBounds --west=140.84472656250003 --north=45.66012730272194 --south=44.10336537791152 --east=143.04199218750003 --zoom=14 --workers=4 --local-scheduler

# Center of Hokkaido
python CSReliefMap.py generateImageByBounds --west=141.67968750000003 --north=44.35527821160296 --south=42.76314586689494 --east=143.87695312500003 --zoom=14 --workers=4 --local-scheduler

# East of Hokkaido
python CSReliefMap.py generateImageByBounds --west=143.94287109375003 --north=44.38669150215206 --south=42.79540065303723 --east=146.14013671875003 --zoom=14 --workers=4 --local-scheduler

# Mid-South of Hokkaido
python CSReliefMap.py generateImageByBounds --west=141.74560546875003 --north=43.51668853502909 --south=41.902277040963696 --east=143.94287109375003 --zoom=14 --workers=4 --local-scheduler

# West-South of Hokkaido
python CSReliefMap.py generateImageByBounds --west=139.32861328125003 --north=43.50075243569041 --south=41.32732632036624 --east=141.65771484375003 --zoom=14 --workers=4 --local-scheduler

# around Tokachi
python CSReliefMap.py generateImageByBounds --west=142.74810791015625 --north=43.25320494908846 --south=42.21224516288584 --east=143.72589111328125 --zoom=14 --workers=4 --local-scheduler

# around Tokyo
python CSReliefMap.py generateImageByBounds --west=139.559326171875 --north=35.77994251888403 --south=35.36217605914681 --east=140.2569580078125 --zoom=15 --workers=4 --local-scheduler

# generate each tile
python CSReliefMap.py generateImageCurvature --x 29139 --y 12936 --z 15 --local-scheduler

# generate each tile (slope only)
python CSReliefMap.py generateImageSlope --x 29139 --y 12936 --z 15 --local-scheduler

# generate each tile (curvature only)
python CSReliefMap.py generateImageCSReliefMap --x 29139 --y 12936 --z 15 --local-scheduler

```

##### Utility for coordinate picking
 
 - [leaflet-areaselect](http://heyman.github.io/leaflet-areaselect/example/)

#### generate by japanese 1-level mesh(1次メッシュ)

```python
python CSReliefMap.py generateImageByMeshCodes --meshcodes "[6341,6342]"
```

#### generated result

## License

### Software

MIT License  
Copyright (c) 2017 Kunihiko Miyoshi

### Generated tile images

<a rel="license" href="http://creativecommons.org/licenses/by/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by/4.0/">Creative Commons Attribution 4.0 International License</a>.

### Elevation data

[国土地理院 標高タイル（基盤地図情報数値標高モデル）](http://maps.gsi.go.jp/development/ichiran.html)

