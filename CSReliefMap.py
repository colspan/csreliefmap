#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
CS Relief Map Tile Image Generator
"""

from urlparse import urlparse
import os
from math import cos, tan, radians, pi, log
from colorsys import hls_to_rgb

import requests
import numpy as np
from PIL import Image, ImageDraw, ImageChops
from matplotlib import pyplot as plt, cm, colors

import luigi


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


class loadDem(DownloadTile):
    """
    標高タイル（基盤地図情報数値標高モデル）
    https://maps.gsi.go.jp/development/ichiran.html
    """
    baseName = "dem"
    baseUrl = None

    def __init__(self, *args, **kwargs):
        super(loadDem, self).__init__(*args, **kwargs)
        if self.z == 15:
            # 15
            self.baseUrl = "http://cyberjapandata.gsi.go.jp/xyz/dem5a/{z}/{x}/{y}.txt"
        elif 0 <= self.z <= 14:
            # 0-14
            self.baseUrl = "http://cyberjapandata.gsi.go.jp/xyz/dem/{z}/{x}/{y}.txt"
        else:
            raise


class calcDemSlope(luigi.Task):
    # http://www.spatialanalysisonline.com/HTML/index.html?profiles_and_curvature.htm
    x = luigi.IntParameter()
    y = luigi.IntParameter()
    z = luigi.IntParameter()
    xRange = luigi.ListParameter(default=[-1, 1000000000])
    yRange = luigi.ListParameter(default=[-1, 1000000000])
    folder_name = "demSlope"

    def requires(self):
        def chooseTask(x, y):
            if(x < min(self.xRange) or x >= max(self.xRange)):
                return None
            if(y < min(self.yRange) or y >= max(self.yRange)):
                return None
            return loadDem(x=x, y=y, z=self.z)
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
            self.folder_name,
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
        with self.output().open("w") as output_f:
            np.savetxt(output_f, grd)


class calcDemCurvature(calcDemSlope):
    folder_name = "demCurvature"

    def run(self):
        # http://www.spatialanalysisonline.com/HTML/index.html?profiles_and_curvature.htm
        def dz_dx(a, x, y):
            return (a[x + 1, y] - a[x - 1, y]) / 2

        def dz_dy(a, x, y):
            return (a[x, y + 1] - a[x, y - 1]) / 2

        def dzdz_dxdy(a, x, y):
            return (a[x + 1, y + 1] - a[x - 1, y + 1] -
                    a[x + 1, y - 1] + a[x - 1, y - 1]) / (2 * 2)

        def dzdz_dxdx(a, x, y):
            return (a[x + 1, y] - 2 * a[x, y] + a[x - 1, y])

        def dzdz_dydy(a, x, y):
            return (a[x, y + 1] - 2 * a[x, y] + a[x, y - 1])

        def curvature(a, x, y):
            p = dz_dx(a, x, y)**2 + dz_dy(a, x, y)**2
            q = p + 1
            return (dzdz_dxdx(a, x, y) * (dz_dx(a, x, y)**2) + 2 * dzdz_dxdy(a, x, y) * dz_dx(a, x, y) * dz_dy(a, x, y) + dzdz_dydy(a, x, y) * (dz_dy(a, x, y)**2)) / (p * q**1.5 + 1)

        combined_tile = np.nan_to_num(self._combine_tiles())
        cur = np.empty((256, 256))
        for i in range(256):
            for j in range(256):
                cur[i, j] = curvature(combined_tile, 256 + i, 256 + j)

        with self.output().open("w") as output_f:
            np.savetxt(output_f, np.nan_to_num(cur))


class generateImageSlope(luigi.Task):
    x = luigi.IntParameter()
    y = luigi.IntParameter()
    z = luigi.IntParameter()
    folder_name = "demSlope"
    cmap_name = "Reds"
    cmap_range = [0, 70]
    abs_filter = True

    def __init__(self, *args, **kwargs):
        super(generateImageSlope, self).__init__(*args, **kwargs)
        self.cmapper = self.get_color_mapper()

    def get_color_mapper(self):
        cmapper = cm.ScalarMappable(norm=colors.SymLogNorm(linthresh=0.03, linscale=0.03,
                                                           vmin=min(self.cmap_range), vmax=max(self.cmap_range), clip=True), cmap=plt.get_cmap(self.cmap_name))
        return cmapper

    def output(self):
        output_file = "./var/{}/{}/{}/{}.{}".format(
            self.folder_name,
            self.z,
            self.x,
            self.y,
            "png"
        )
        return luigi.LocalTarget(output_file)

    def requires(self):
        return calcDemSlope(x=self.x, y=self.y, z=self.z)

    def get_color(self, value):
        color = self.cmapper.to_rgba(value)
        return tuple([int(256 * x) for x in color])

    def run(self):
        size_x, size_y = (256, 256)

        with self.input().open("r") as input_f:
            data = np.loadtxt(input_f)

        if self.abs_filter:
            data = np.abs(data)

        d_max = np.max(data)
        d_min = np.min(data)
        print (d_max, d_min)
        colored_data = np.uint8(self.cmapper.to_rgba(data) * 255)
        colored_data[:, :, 3] = 255
        img = Image.fromarray(colored_data)
        img.save(self.output().fn, 'PNG')


class generateImageCurvature(generateImageSlope):
    folder_name = "demCurvature"
    cmap_name = "Blues"
    cmap_range = [-150, 150]
    abs_filter = False

    def get_color_mapper(self):
        cmapper = cm.ScalarMappable(norm=colors.SymLogNorm(linthresh=0.015, linscale=0.03,
                                                           vmin=min(self.cmap_range), vmax=max(self.cmap_range), clip=True), cmap=plt.get_cmap(self.cmap_name))
        return cmapper

    def requires(self):
        return calcDemCurvature(x=self.x, y=self.y, z=self.z)


class generateImageCSReliefMap(luigi.Task):
    x = luigi.IntParameter()
    y = luigi.IntParameter()
    z = luigi.IntParameter()
    folder_name = "CSReliefMap"

    def requires(self):
        return [
            generateImageSlope(x=self.x, y=self.y, z=self.z),
            generateImageCurvature(x=self.x, y=self.y, z=self.z),
        ]

    def output(self):
        output_file = "./var/{}/{}/{}/{}.{}".format(
            self.folder_name,
            self.z,
            self.x,
            self.y,
            "png"
        )
        return luigi.LocalTarget(output_file)

    def run(self):
        size_x, size_y = (256, 256)

        output_img = Image.new('RGBA', (size_x, size_y), (255, 255, 255, 255))
        for source in self.input():
            input_img = Image.open(source.fn)
            output_img = ImageChops.multiply(output_img, input_img)

        with self.output().open("wb") as output_f:
            output_img.save(output_f, 'PNG')


class generateImageBounds(luigi.WrapperTask):
    """
    Schedule Download Tasks
    """
    west = luigi.FloatParameter()
    north = luigi.FloatParameter()
    south = luigi.FloatParameter()
    east = luigi.FloatParameter()
    zoom = luigi.IntParameter()
    targetTask = luigi.TaskParameter(default=generateImageCSReliefMap)

    def requires(self):
        """
        scheduling tasks
        """

        candidateTasks = [generateImageCSReliefMap,
                          generateImageCurvature, generateImageSlope]
        if not self.targetTask in candidateTasks:
            raise

        edge_nw_x, edge_nw_y, _, _ = deg_to_num(
            self.north, self.west, self.zoom)
        edge_se_x, edge_se_y, _, _ = deg_to_num(
            self.south, self.east, self.zoom)
        #xRange = [edge_nw_x, edge_se_x]
        #yRange = [edge_nw_y, edge_se_y]
        print deg_to_num(self.north, self.west, self.zoom) + deg_to_num(self.south, self.east, self.zoom)
        for tile_x in range(edge_nw_x, edge_se_x + 1):
            for tile_y in range(edge_nw_y, edge_se_y + 1):
                yield self.targetTask(x=tile_x, y=tile_y, z=self.zoom)


if __name__ == "__main__":
    luigi.run()
