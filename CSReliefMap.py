#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Map Tile Downloader based on Luigi
"""

from urlparse import urlparse
import os
from math import cos, tan, radians, pi, log

import numpy as np

import luigi
import requests


def deg_to_num(lat_deg, lon_deg, zoom):
    """
    degree to num
    """
    lat_rad = radians(lat_deg)
    n = 2.0 ** zoom
    xtile_f = (lon_deg + 180.0) / 360.0 * n
    ytile_f = (1.0 - log(tan(lat_rad) + (1 / cos(lat_rad))) / pi) / 2.0 * n
    xtile = int(xtile_f)
    ytile = int(ytile_f)
    pos_x = int((xtile_f - xtile) * 256)
    pos_y = int((ytile_f - ytile) * 256)
    return (xtile, ytile, pos_x, pos_y)


class DownloadTile(luigi.Task):
    """
    Download a Tile
    """
    baseUrl = luigi.Parameter()
    baseName = luigi.Parameter()
    x = luigi.IntParameter()
    y = luigi.IntParameter()
    z = luigi.IntParameter()

    def output(self):
        """
        define output
        """
        extension = os.path.splitext(urlparse(self.baseUrl).path)[
            1].replace(".", "")
        output_file = "./var/{}/{}/{}/{}.{}".format(
            self.baseName,
            self.z,
            self.x,
            self.y,
            extension
        )
        return luigi.LocalTarget(output_file)

    def run(self):
        """
        download a tile
        """
        url = self.baseUrl.format(**{"x": self.x, "y": self.y, "z": self.z})
        req = requests.get(url, stream=True)
        with self.output().open("wb") as f_out:
            for chunk in req.iter_content(chunk_size=1024):
                f_out.write(chunk)


class DownloadBounds(luigi.WrapperTask):
    """
    Schedule Download Tasks
    """
    baseUrl = luigi.Parameter()
    baseName = luigi.Parameter(default="output")
    west = luigi.FloatParameter()
    north = luigi.FloatParameter()
    south = luigi.FloatParameter()
    east = luigi.FloatParameter()
    zoom = luigi.IntParameter()

    def requires(self):
        """
        scheduling tasks
        """
        edge_nw_x, edge_nw_y, _, _ = deg_to_num(
            self.north, self.west, self.zoom)
        edge_se_x, edge_se_y, _, _ = deg_to_num(
            self.south, self.east, self.zoom)
        print deg_to_num(self.north, self.west, self.zoom) + deg_to_num(self.south, self.east, self.zoom)
        for tile_x in range(edge_nw_x, edge_se_x + 1):
            for tile_y in range(edge_nw_y, edge_se_y + 1):
                print "scheduling z:{} x:{} y:{}".format(self.zoom, tile_x, tile_y)
                yield DownloadTile(self.baseUrl, self.baseName, tile_x, tile_y, self.zoom)


class loadDem5b(DownloadTile):
    baseUrl = "http://cyberjapandata.gsi.go.jp/xyz/dem5b/{z}/{x}/{y}.txt"
    baseName = "dem5b"


class calcDem5bSlope(luigi.Task):
    # http://www.spatialanalysisonline.com/HTML/index.html?profiles_and_curvature.htm
    x = luigi.IntParameter()
    y = luigi.IntParameter()
    z = luigi.IntParameter()
    xRange = luigi.ListParameter()
    yRange = luigi.ListParameter()

    def requires(self):
        def chooseTask(x, y):
            if(x < min(self.xRange) or x >= max(self.xRange)):
                return None
            if(y < min(self.yRange) or y >= max(self.yRange)):
                return None
            return loadDem5b(x=x, y=y, z=self.z)
        self.taskList = [
            [chooseTask(x=self.x - 1, y=self.y - 1),
             chooseTask(x=self.x, y=self.y - 1),
             chooseTask(x=self.x + 1, y=self.y - 1)],
            [chooseTask(x=self.x - 1, y=self.y),
             chooseTask(x=self.x, y=self.y),
             chooseTask(x=self.x + 1, y=self.y)],
            [chooseTask(x=self.x - 1, y=self.y + 1),
             chooseTask(x=self.x, y=self.y + 1),
             chooseTask(x=self.x + 1, y=self.y + 1)]
        ]
        return filter(lambda x: x != None, self.taskList)

    def output(self):
        """
        define output
        """
        output_file = "./var/{}/{}/{}/{}.{}".format(
            "dem5bSlope",
            self.z,
            self.x,
            self.y,
            "txt"
        )
        return luigi.LocalTarget(output_file)

    def _load_tile_txt(self, f):
        tile = np.empty((256, 256))
        try:
            for i, line in enumerate(f):
                tile[i, :] = np.array(
                    [float(x) if x != "e" else np.NAN for x in line.strip().split(",")])
        except:
            tile[:] = np.NAN
        return tile

    def _combine_tiles(self):
        combinedTile = np.empty((256 * 3, 256 * 3))
        for i in range(3):
            for j in range(3):
                with self.taskList[i][j].output().open("r") as input_f:
                    tile = self._load_tile_txt(input_f)
                    combinedTile[256 * i:256 *
                                 (i + 1), 256 * j:256 * (j + 1)] = tile
        return combinedTile

    def run(self):
        combinedTile = self._combine_tiles()
        # print combinedTile
        grd = np.gradient(np.nan_to_num(combinedTile))[
            0][256:256 * 2, 256:256 * 2]
        print grd


if __name__ == "__main__":
    luigi.run()
