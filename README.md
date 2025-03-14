# CS Relief Map Tile Image Generator

An one stop tile image generator of CS Relief Map written in Python, integrated with [Luigi](https://github.com/spotify/luigi) and [numpy](http://www.numpy.org/).

![Example Image (--x 29016 --y 12938 --z 15)](example.png)

## What is CS Relief Map (CS立体図) ?

The "*CS Relief Map*" (CS立体図) is a terrain visualization method with high visibility of **C**urvature and **S**lope. This was proposed by [Nagano prefecture forest center (長野県林業総合センター)](https://www.pref.nagano.lg.jp/ringyosogo/).

This implementation is based on these articles.

 - [曲率と傾斜による立体図法（CS立体図）を用いた地形判読](https://www.jstage.jst.go.jp/article/jjfe/56/2/56_KJ00009647426/_pdf)
 - [数値地形データを用いた「微地形図」の作成方法](http://www.pref.nagano.lg.jp/ringyosogo/seika/documents/bichikei.pdf)
 - [QGISで「CS立体図」の「立体図」を作ってみた](http://koutochas.seesaa.net/article/444171690.html)

### Setup

```sh
sudo pip install -r requirements.txt
```

### Examples

#### generate by coordinate

```sh
# around Tokachi
python CSReliefMap.py GenerateImageByBounds --west 142.74810791015625 --north 43.25320494908846 --south 42.21224516288584 --east 143.72589111328125 --zoom 14 --workers 4 --local-scheduler

# around Tokyo
python CSReliefMap.py GenerateImageByBounds --west 139.559326171875 --north 35.77994251888403 --south 35.36217605914681 --east 140.2569580078125 --zoom 15 --workers 4 --local-scheduler

# generate each tile (curvature only)
python CSReliefMap.py GenerateImageCurvature --x 29016 --y 12938 --z 15 --local-scheduler

# generate each tile (slope only)
python CSReliefMap.py GenerateImageSlope --x 29016 --y 12938 --z 15 --local-scheduler

# generate each tile
python CSReliefMap.py GenerateImageCSReliefMap --x 29016 --y 12938 --z 15 --local-scheduler
```

##### Resizing large image to small image  (experimental)
```sh
# single tile
python CSReliefMap.py resizeTileImage --x 3641 --y 1616 --z 12 --sourceZ 14 --sourceTask generateImageCSReliefMap --workers 4 --local-scheduler 

# around Tokyo
python CSReliefMap.py GenerateResizedImageByBounds --west 139.559326171875 --north 35.77994251888403 --south 35.36217605914681 --east 140.2569580078125 --zoom 10 --workers 4 --local-scheduler


```

##### Utility for coordinate picking
 
 - [leaflet-areaselect](http://heyman.github.io/leaflet-areaselect/example/)

#### generate by japanese level-1 mesh(1次メッシュ)

```sh
python CSReliefMap.py GenerateImageByMeshCodes --meshcodes "[6341,6342]" --zoom 14 --workers 4 --local-scheduler

# resized tile map based on zoom 14 (experimental)
python CSReliefMap.py GenerateImageByMeshCodes --targetTask generateResizedImageByBounds --meshcodes "[6341,6342]" --zoom 10 --workers 4 --local-scheduler


```

#### generated result

Generated results are available at [CS Relief Map Web Viewer](http://maps.colspan.net/csrelief/v2/).

## License

### Software

MIT License  
Copyright (c) 2017 Kunihiko Miyoshi

### Generated tile images

<a rel="license" href="http://creativecommons.org/licenses/by/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by/4.0/">Creative Commons Attribution 4.0 International License</a>.

この地図の作成にあたっては、国土地理院長の承認を得て、同院発行の基盤地図情報を使用した。(承認番号 平29情使、 第392号)

### Elevation data source

 - [国土地理院 標高タイル（基盤地図情報数値標高モデル）](http://maps.gsi.go.jp/development/ichiran.html)
 - [data spec](https://maps.gsi.go.jp/development/demtile.html)

Currently the data downloader is available for Japan only. But this project is easily extendable for any regions, any countries.
