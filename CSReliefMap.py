#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
CS Relief Map Tile Image Generator
"""


import logging
import luigi
import numpy as np
import os
import requests

from math import cos, tan, radians, pi, log
from matplotlib import pyplot as plt, cm, colors
from PIL import Image, ImageChops
from urllib.parse import urlparse


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
        extension = os.path.splitext(urlparse(self.baseUrl).path)[1].replace(".", "")
        output_file = "./var/{}/{}/{}/{}.{}".format(
            self.baseName, self.z, self.x, self.y, extension
        )
        return luigi.LocalTarget(output_file, format=luigi.format.Nop)

    def run(self):
        """
        download a tile
        """
        url = self.baseUrl.format(**{"x": self.x, "y": self.y, "z": self.z})
        req = requests.get(url, stream=True)
        with self.output().open("wb") as f_out:
            for chunk in req.iter_content(chunk_size=1024):
                f_out.write(chunk)


class LoadDem(DownloadTile):
    """
    標高タイル（基盤地図情報数値標高モデル）
    https://maps.gsi.go.jp/development/ichiran.html
    """

    baseName = "dem"
    baseUrl = None

    def __init__(self, *args, **kwargs):
        super(LoadDem, self).__init__(*args, **kwargs)
        if self.z == 15:
            # 15
            self.baseUrl = "http://cyberjapandata.gsi.go.jp/xyz/dem5a/{z}/{x}/{y}.txt"
        elif 0 <= self.z <= 14:
            # 0-14
            self.baseUrl = "http://cyberjapandata.gsi.go.jp/xyz/dem/{z}/{x}/{y}.txt"
        else:
            raise Exception("unsupported z value : {}".format(self.z))

    def load_data(self):
        tile = np.empty((256, 256))
        with open(self.output().fn, "r") as f:
            try:
                for i, line in enumerate(f):
                    tile[i, :] = np.array(
                        [
                            float(x) if x != "e" else np.nan # 水域は np.nan にする
                            for x in line.strip().split(",")
                        ]
                    )
            except Exception as e:
                logging.warning(
                    "skipping load {} and compensating with `np.nan`".format(
                        self.output().path
                    )
                )
                tile[:] = np.nan # 水域は np.nan にする
        return tile


class CalcDemSlope(luigi.Task):
    # http://www.spatialanalysisonline.com/HTML/index.html?profiles_and_curvature.htm
    x = luigi.IntParameter()
    y = luigi.IntParameter()
    z = luigi.IntParameter()
    xRange = luigi.ListParameter(default=[-1, 1000000000])
    yRange = luigi.ListParameter(default=[-1, 1000000000])
    folder_name = "demSlope"

    def requires(self):
        def choose_task(x, y):
            if x < min(self.xRange) or x >= max(self.xRange):
                return None
            if y < min(self.yRange) or y >= max(self.yRange):
                return None
            return LoadDem(x=x, y=y, z=self.z)

        self.taskList = [
            [
                choose_task(x=self.x - 1, y=self.y - 1),
                choose_task(x=self.x, y=self.y - 1),
                choose_task(x=self.x + 1, y=self.y - 1),
            ],
            [
                choose_task(x=self.x - 1, y=self.y),
                choose_task(x=self.x, y=self.y),
                choose_task(x=self.x + 1, y=self.y),
            ],
            [
                choose_task(x=self.x - 1, y=self.y + 1),
                choose_task(x=self.x, y=self.y + 1),
                choose_task(x=self.x + 1, y=self.y + 1),
            ],
        ]
        return filter(lambda x: x != None, self.taskList)

    def output(self):
        """
        define output
        """
        output_file = "./var/{}/{}/{}/{}.{}".format(
            self.folder_name, self.z, self.x, self.y, "npy"
        )
        return luigi.LocalTarget(output_file, format=luigi.format.Nop)

    def _combine_tiles(self):
        combinedTile = np.empty((256 * 3, 256 * 3))
        for i in range(3):
            for j in range(3):
                if self.taskList[i][j] != None:
                    tile = self.taskList[i][j].load_data()
                    combinedTile[
                        256 * i : 256 * (i + 1), 256 * j : 256 * (j + 1)
                    ] = tile
        return combinedTile

    def run(self):
        combinedTile = self._combine_tiles()
        # print combinedTile
        grd = np.gradient(np.nan_to_num(combinedTile))
        grd_x = grd[0][256 : 256 * 2, 256 : 256 * 2]
        grd_y = grd[1][256 : 256 * 2, 256 : 256 * 2]

        with self.output().open("w") as output_f:
            np.save(output_f, np.abs(grd_x) + np.abs(grd_y))


class CalcDemCurvature(CalcDemSlope):
    folder_name = "demCurvature"

    def run(self):
        # http://www.spatialanalysisonline.com/HTML/index.html?profiles_and_curvature.htm
        def dz_dx(a, x, y):
            return (a[x + 1, y] - a[x - 1, y]) / 2

        def dz_dy(a, x, y):
            return (a[x, y + 1] - a[x, y - 1]) / 2

        def dzdz_dxdy(a, x, y):
            return (
                a[x + 1, y + 1] - a[x - 1, y + 1] - a[x + 1, y - 1] + a[x - 1, y - 1]
            ) / (2 * 2)

        def dzdz_dxdx(a, x, y):
            return a[x + 1, y] - 2 * a[x, y] + a[x - 1, y]

        def dzdz_dydy(a, x, y):
            return a[x, y + 1] - 2 * a[x, y] + a[x, y - 1]

        def curvature(a, x, y):
            p = dz_dx(a, x, y) ** 2 + dz_dy(a, x, y) ** 2
            q = p + 1
            return (
                dzdz_dxdx(a, x, y) * (dz_dx(a, x, y) ** 2)
                + 2 * dzdz_dxdy(a, x, y) * dz_dx(a, x, y) * dz_dy(a, x, y)
                + dzdz_dydy(a, x, y) * (dz_dy(a, x, y) ** 2)
            ) / (p * q ** 1.5 + 1)

        combined_tile = np.nan_to_num(self._combine_tiles())
        cur = np.empty((256, 256))
        for i in range(256):
            for j in range(256):
                cur[i, j] = curvature(combined_tile, 256 + i, 256 + j)

        with self.output().open("w") as output_f:
            np.save(output_f, np.nan_to_num(cur))


def generate_height_map(data, cmap_name="bone_r", cmap_range=[0, 1000000]):
    cmapper = cm.ScalarMappable(
        norm=colors.SymLogNorm(
            linthresh=1,
            linscale=0.01,
            base=np.e,
            vmin=min(cmap_range),
            vmax=max(cmap_range),
            clip=True,
        ),
        cmap=plt.get_cmap(cmap_name),
    )
    colored_data = np.uint8(cmapper.to_rgba(data) * 255)
    colored_data[:, :, 3] = 255
    colored_data = np.uint8(colored_data / 2 + 127)
    img = Image.fromarray(colored_data)
    return img


def generate_sea_map(data, color=(196, 218, 255, 255)):
    colored_data = np.zeros((256, 256, 4), dtype=np.uint8)
    colored_data *= 255
    colored_data[np.isnan(data), :] = np.array(color)
    img = Image.fromarray(colored_data)
    return img


class GenerateSeaMap(luigi.Task):
    x = luigi.IntParameter()
    y = luigi.IntParameter()
    z = luigi.IntParameter()
    folder_name = "imgSeaMap"
    color = luigi.TupleParameter(default=(196, 218, 255, 255))

    def output(self):
        output_file = "./var/{}/{}/{}/{}.{}".format(
            self.folder_name, self.z, self.x, self.y, "png"
        )
        return luigi.LocalTarget(output_file, format=luigi.format.Nop)

    def requires(self):
        return LoadDem(x=self.x, y=self.y, z=self.z)

    def run(self):
        data = self.requires().load_data()
        img = generate_sea_map(data, color=self.color)

        with self.output().open("wb") as output_f:
            img.save(output_f, "PNG")


def generate_image_slope(data, cmap_name="Blues", cmap_range=[0, 80]):
    cmapper = cm.ScalarMappable(
        norm=colors.SymLogNorm(
            linthresh=0.5,
            linscale=0.5,
            base=np.e,
            vmin=min(cmap_range),
            vmax=max(cmap_range),
            clip=True,
        ),
        cmap=plt.get_cmap(cmap_name),
    )
    colored_data = np.uint8(cmapper.to_rgba(np.abs(data)) * 255)
    colored_data[:, :, 3] = 255
    img = Image.fromarray(colored_data)
    return img


class GenerateImageSlope(luigi.Task):
    x = luigi.IntParameter()
    y = luigi.IntParameter()
    z = luigi.IntParameter()
    folder_name = "imgDemSlope"
    cmap_name = "Blues"
    cmap_range = [0, 25]

    def output(self):
        output_file = "./var/{}/{}/{}/{}.{}".format(
            self.folder_name, self.z, self.x, self.y, "png"
        )
        return luigi.LocalTarget(output_file, format=luigi.format.Nop)

    def requires(self):
        return CalcDemSlope(x=self.x, y=self.y, z=self.z)

    def run(self):
        size_x, size_y = (256, 256)

        with self.input().open("r") as input_f:
            data = np.load(input_f)

        d_max = np.max(data)
        d_min = np.min(data)
        print(d_max, d_min)
        img = generate_image_slope(
            data, cmap_name=self.cmap_name, cmap_range=self.cmap_range
        )

        with self.output().open("wb") as output_f:
            img.save(output_f, "PNG")


def generate_image_curvature(data, cmap_name="Greens", cmap_range=[-0.07, 0.07]):
    cmapper = cm.ScalarMappable(
        norm=colors.SymLogNorm(
            linthresh=0.5,
            linscale=0.5,
            vmin=min(cmap_range),
            vmax=max(cmap_range),
            clip=True,
        ),
        cmap=plt.get_cmap(cmap_name),
    )
    colored_data = np.uint8(cmapper.to_rgba(data) * 255)
    colored_data[:, :, 3] = 255
    img = Image.fromarray(colored_data)
    return img


class GenerateImageCurvature(GenerateImageSlope):
    folder_name = "imgDemCurvature"
    cmap_name = "Greens"
    cmap_range = [-0.07, 0.07]

    def requires(self):
        return CalcDemCurvature(x=self.x, y=self.y, z=self.z)

    def run(self):
        size_x, size_y = (256, 256)

        with self.input().open("r") as input_f:
            data = np.load(input_f)

        d_max = np.max(data)
        d_min = np.min(data)
        print(d_max, d_min)
        img = generate_image_curvature(
            data, cmap_name=self.cmap_name, cmap_range=self.cmap_range
        )

        with self.output().open("wb") as output_f:
            img.save(output_f, "PNG")


class GenerateImageCSReliefMap(luigi.Task):
    x = luigi.IntParameter()
    y = luigi.IntParameter()
    z = luigi.IntParameter()
    folder_name = "imgCSReliefMap"

    def requires(self):
        return [
            LoadDem(x=self.x, y=self.y, z=self.z),
            CalcDemSlope(x=self.x, y=self.y, z=self.z),
            CalcDemCurvature(x=self.x, y=self.y, z=self.z),
        ]

    def output(self):
        output_file = "./var/{}/{}/{}/{}.{}".format(
            self.folder_name, self.z, self.x, self.y, "png"
        )
        return luigi.LocalTarget(output_file, format=luigi.format.Nop)

    def run(self):
        size_x, size_y = (256, 256)

        imgs = []

        data_dem = self.requires()[0].load_data()
        # Height Map
        imgs.append(generate_height_map(data_dem))

        # Slope
        with self.input()[1].open("r") as input_f:
            data_slope = np.load(input_f)
        imgs.append(generate_image_slope(data_slope))

        # Curvature
        with self.input()[2].open("r") as input_f:
            data_curvature = np.load(input_f)
        imgs.append(generate_image_curvature(data_curvature))

        output_img = Image.new("RGBA", (size_x, size_y), (255, 255, 255, 255))
        for img in imgs:
            output_img = ImageChops.multiply(output_img, img)

        # Sea Map
        sea_img = generate_sea_map(data_dem)
        output_img = ImageChops.composite(sea_img, output_img, sea_img)

        with self.output().open("wb") as output_f:
            output_img.save(output_f, "PNG")


class ResizeTileImage(luigi.Task):
    x = luigi.IntParameter()
    y = luigi.IntParameter()
    z = luigi.IntParameter()
    ignore_no_image = luigi.BoolParameter(default=False)
    fill_image = luigi.Parameter(default=None)
    max_search_z = luigi.IntParameter(default=14)
    folder_name = luigi.Parameter(default="resized")
    sourceZ = luigi.IntParameter()
    sourceTask = luigi.TaskParameter(default=GenerateImageCSReliefMap)

    def requires(self):
        def choose_task(x, y, z):
            if z == self.sourceZ:
                return self.sourceTask(x=x, y=y, z=z)
            else:
                return ResizeTileImage(
                    x=x,
                    y=y,
                    z=z,
                    ignore_no_image=self.ignore_no_image,
                    fill_image=self.fill_image,
                    max_search_z=self.max_search_z,
                    folder_name=self.folder_name,
                    sourceTask=self.sourceTask,
                    sourceZ=self.sourceZ,
                )

        x = self.x * 2
        y = self.y * 2
        z = self.z + 1
        self.task_list = [
            [
                choose_task(x=x - 1, y=y - 1, z=z),
                choose_task(x=x, y=y - 1, z=z),
                choose_task(x=x + 1, y=y - 1, z=z),
                choose_task(x=x + 2, y=y - 1, z=z),
            ],
            [
                choose_task(x=x - 1, y=y, z=z),
                choose_task(x=x, y=y, z=z),
                choose_task(x=x + 1, y=y, z=z),
                choose_task(x=x + 2, y=y, z=z),
            ],
            [
                choose_task(x=x - 1, y=y + 1, z=z),
                choose_task(x=x, y=y + 1, z=z),
                choose_task(x=x + 1, y=y + 1, z=z),
                choose_task(x=x + 2, y=y + 1, z=z),
            ],
            [
                choose_task(x=x - 1, y=y + 2, z=z),
                choose_task(x=x, y=y + 2, z=z),
                choose_task(x=x + 1, y=y + 2, z=z),
                choose_task(x=x + 2, y=y + 2, z=z),
            ],
        ]
        if self.ignore_no_image or self.max_search_z < z:
            return None
        else:
            return filter(lambda x: x != None, self.task_list)

    def output(self):
        """
        define output
        """
        output_file = "./var/{}/{}/{}/{}.{}".format(
            self.folder_name, self.z, self.x, self.y, "png"
        )
        return luigi.LocalTarget(output_file, format=luigi.format.Nop)

    def _combine_tiles(self):
        combined_tile = Image.new("RGBA", (256 * 4, 256 * 4), (255, 255, 255, 255))
        for i in range(4):
            for j in range(4):
                target = self.task_list[i][j]
                if self.ignore_no_image and not target.complete():
                    input_img = Image.open(self.fill_image)
                else:
                    input_img = Image.open(target.output().fn)

                combined_tile.paste(input_img, (256 * j, 256 * i))
        return combined_tile

    def run(self):
        combined_tile = self._combine_tiles()
        combined_tile.thumbnail((512, 512), Image.ANTIALIAS)
        cropped_image = combined_tile.crop((128, 128, 384, 384))
        with self.output().open("wb") as output_f:
            cropped_image.save(output_f, "PNG")


class GenerateImageByBounds(luigi.WrapperTask):
    """
    Schedule Download Tasks
    """

    west = luigi.FloatParameter()
    north = luigi.FloatParameter()
    south = luigi.FloatParameter()
    east = luigi.FloatParameter()
    zoom = luigi.IntParameter()
    targetTask = luigi.TaskParameter(default=GenerateImageCSReliefMap)

    def requires(self):
        """
        scheduling tasks
        """

        candidateTasks = [
            GenerateImageCSReliefMap,
            GenerateImageCurvature,
            GenerateImageSlope,
        ]
        if not self.targetTask in candidateTasks:
            raise Exception("no target task")

        edge_nw_x, edge_nw_y, _, _ = deg_to_num(self.north, self.west, self.zoom)
        edge_se_x, edge_se_y, _, _ = deg_to_num(self.south, self.east, self.zoom)
        # xRange = [edge_nw_x, edge_se_x]
        # yRange = [edge_nw_y, edge_se_y]
        print(
            deg_to_num(self.north, self.west, self.zoom)
            + deg_to_num(self.south, self.east, self.zoom)
        )
        for tile_x in range(edge_nw_x - 3, edge_se_x + 3):
            for tile_y in range(edge_nw_y - 3, edge_se_y + 3):
                yield self.targetTask(x=tile_x, y=tile_y, z=self.zoom)


class GenerateResizedImageByBounds(luigi.WrapperTask):
    """
    Schedule Download Tasks
    """

    west = luigi.FloatParameter()
    north = luigi.FloatParameter()
    south = luigi.FloatParameter()
    east = luigi.FloatParameter()
    zoom = luigi.IntParameter()
    targetTask = luigi.TaskParameter(default=GenerateImageCSReliefMap)
    ignore_no_image = luigi.BoolParameter(default=False)
    fill_image = luigi.Parameter(default=None)
    max_search_z = luigi.IntParameter(default=14)
    folder_name = luigi.Parameter(default="resized")
    sourceZoom = luigi.IntParameter(default=14)

    def requires(self):
        """
        scheduling tasks
        """

        edge_nw_x, edge_nw_y, _, _ = deg_to_num(self.north, self.west, self.zoom)
        edge_se_x, edge_se_y, _, _ = deg_to_num(self.south, self.east, self.zoom)
        print(
            deg_to_num(self.north, self.west, self.zoom)
            + deg_to_num(self.south, self.east, self.zoom)
        )
        for tile_x in range(edge_nw_x - 3, edge_se_x + 3):
            for tile_y in range(edge_nw_y - 3, edge_se_y + 3):
                yield ResizeTileImage(
                    x=tile_x,
                    y=tile_y,
                    z=self.zoom,
                    sourceTask=self.targetTask,
                    ignore_no_image=self.ignore_no_image,
                    fill_image=self.fill_image,
                    max_search_z=self.max_search_z,
                    folder_name=self.folder_name,
                    sourceZ=self.sourceZoom,
                )


def meshcode_to_latlng(meshcode):
    latitude = float(meshcode[0:2]) / 1.5
    longtitude = float(meshcode[2:4]) + 100.0
    return (latitude, longtitude)


class GenerateImageByMeshCodes(luigi.WrapperTask):
    """
    Schedule Download Tasks
    """

    meshcodes = luigi.ListParameter()
    zoom = luigi.IntParameter()
    targetTask = luigi.TaskParameter(default=GenerateImageByBounds)

    def requires(self):
        for meshcode in self.meshcodes:
            meshcode = str(meshcode)
            south, west = meshcode_to_latlng(meshcode)
            north, east = meshcode_to_latlng(
                "{:02d}{:02d}".format(int(meshcode[0:2]) + 1, int(meshcode[2:4]) + 1)
            )
            yield self.targetTask(
                west=west, north=north, east=east, south=south, zoom=self.zoom
            )


if __name__ == "__main__":
    luigi.run()
