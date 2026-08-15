# -*- coding: utf-8 -*-
"""
Created on Sun Dec 17 19:55:44 2023

@author: 86186
"""

import math
import mujoco
import time
import mujoco.viewer
import random
from openpyxl import Workbook

model=mujoco.MjModel.from_xml_path('C:/Users/Administrator.DESKTOP-I6GS248/Desktop/Mujoco/StableSoft_sim/9bones.xml')
data=mujoco.MjData(model)


def Operate():
    with mujoco.viewer.launch_passive(model,data) as viewer:
        start=time.time()
        i=1
        while viewer.is_running():
            step_start=time.time()
            mujoco.mj_step(model,data)
            if time.time()-start>10:
                data.ctrl[1]=5
                i+=1
            viewer.sync()
            time_until_next_step=model.opt.timestep-(time.time()-step_start)
            if time_until_next_step>0:
                time.sleep(time_until_next_step)
Operate()