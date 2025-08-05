from scenariogeneration import xosc, xodr, prettyprint, ScenarioGenerator
import pyclothoids as pcloth
import os
import numpy as np

class Scenario(ScenarioGenerator):
    def __init__(self):
        super().__init__()

    def road(self, **kwargs):
        # 创建4条道路（十字路口的4个方向）
        road1 = xodr.create_road([xodr.Line(100)], 1, 2, 2)  # 南北方向道路
        road2 = xodr.create_road([xodr.Line(100)], 2, 2, 2)  # 东西方向道路
        road3 = xodr.create_road([xodr.Line(100)], 3, 2, 2)  # 南北方向道路（反向）
        road4 = xodr.create_road([xodr.Line(100)], 4, 2, 2)  # 东西方向道路（反向）

        # 创建圆形交叉口
        jc = xodr.CommonJunctionCreator(100, "cross_roads_junction")

        # 添加道路到交叉口（圆形连接方式）
        jc.add_incoming_road_circular_geometry(road=road1, radius=20, angle=0, road_connection="successor")
        jc.add_incoming_road_circular_geometry(road=road2, radius=20, angle=np.pi/2, road_connection="successor")
        jc.add_incoming_road_circular_geometry(road=road3, radius=20, angle=np.pi, road_connection="successor")
        jc.add_incoming_road_circular_geometry(road=road4, radius=20, angle=3*np.pi/2, road_connection="successor")

        # 设置道路连接关系
        jc.add_connection(1, 2)  # 连接道路1和道路2
        jc.add_connection(2, 3)  # 连接道路2和道路3
        jc.add_connection(3, 4)  # 连接道路3和道路4
        jc.add_connection(4, 1)  # 连接道路4和道路1

        # 创建OpenDRIVE对象
        odr = xodr.OpenDrive("cross_roads")

        # 添加所有道路
        odr.add_road(road1)
        odr.add_road(road2)
        odr.add_road(road3)
        odr.add_road(road4)

        # 添加交叉口
        odr.add_junction_creator(jc)

        # 调整道路和车道
        odr.adjust_roads_and_lanes()

        return odr

if __name__ == "__main__":
    sce = Scenario()
    # 打印生成的XML内容
    prettyprint(sce.road().get_element())
    # 生成场景文件
    sce.generate(".")
