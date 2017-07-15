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

#### generate

```python
# generate each tile
python CSReliefMap.py generateImageCurvature --x 29139 --y 12936 --z 15 --local-scheduler

# generate each tile (slope only)
python CSReliefMap.py generateImageSlope --x 29139 --y 12936 --z 15 --local-scheduler

# generate each tile (curvature only)
python CSReliefMap.py generateImageCSReliefMap --x 29139 --y 12936 --z 15 --local-scheduler

# around Tokachi
python CSReliefMap.py generateImageBounds --west=142.74810791015625 --north=43.25320494908846 --south=42.21224516288584 --east=143.72589111328125 --zoom=14 --workers=4 --local-scheduler

# around Tokyo
python CSReliefMap.py generateImageBounds --west=139.559326171875 --north=35.77994251888403 --south=35.36217605914681 --east=140.2569580078125 --zoom=15 --workers=4 --local-scheduler

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

