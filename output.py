from scenariogeneration import xosc, xodr, prettyprint, ScenarioGenerator
import pyclothoids as pcloth
import os
import numpy as np

class Scenario(ScenarioGenerator):
    def __init__(self):
        super().__init__()

    def road(self, **kwargs):
        # 创建四条道路（均为直线）
        road1 = xodr.create_road([xodr.Line(100)], 1, 2, 2)  # 道路1：100米直线，2车道
        road2 = xodr.create_road([xodr.Line(100)], 2, 2, 2)  # 道路2：100米直线，2车道
        road3 = xodr.create_road([xodr.Line(100)], 3, 2, 2)  # 道路3：100米直线，2车道
        road4 = xodr.create_road([xodr.Line(100)], 4, 2, 2)  # 道路4：100米直线，2车道

        # 创建圆形交叉口（id为100，名称为"cross_junction"）
        jc = xodr.CommonJunctionCreator(100, "cross_junction")

        # 将四条道路连接到交叉口（圆形交叉口）
        # 道路1：角度为0（正东方向）
        jc.add_incoming_road_circular_geometry(road=road1, radius=20, angle=0, road_connection="successor")
        # 道路2：角度为np.pi/2（正北方向）
        jc.add_incoming_road_circular_geometry(road=road2, radius=20, angle=np.pi/2, road_connection="successor")
        # 道路3：角度为np.pi（正西方向）
        jc.add_incoming_road_circular_geometry(road=road3, radius=20, angle=np.pi, road_connection="successor")
        # 道路4：角度为3*np.pi/2（正南方向）
        jc.add_incoming_road_circular_geometry(road=road4, radius=20, angle=3*np.pi/2, road_connection="successor")

        # 设置道路之间的连接关系
        jc.add_connection(1, 2)  # 道路1连接到道路2
        jc.add_connection(2, 3)  # 道路2连接到道路3
        jc.add_connection(3, 4)  # 道路3连接到道路4
        jc.add_connection(4, 1)  # 道路4连接到道路1

        # 创建OpenDRIVE主对象
        odr = xodr.OpenDrive("cross_road")

        # 添加道路到OpenDRIVE
        odr.add_road(road1)
        odr.add_road(road2)
        odr.add_road(road3)
        odr.add_road(road4)

        # 添加交叉口到OpenDRIVE
        odr.add_junction_creator(jc)

        # 调整道路和车道
        odr.adjust_roads_and_lanes()

        return odr

if __name__ == "__main__":
    sce = Scenario()
    prettyprint(sce.road().get_element())  # 打印XML内容
    sce.generate(".")  # 生成并保存场景文件
