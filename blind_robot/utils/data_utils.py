dtype_data = [
    ("idnum", int),  # 00.
    ("actx", float),  # 01.  (tcp (tool center point) position (3):
    #       x,y,z in absolute world coordinates)
    ("acty", float),  # 02.
    ("actz", float),  # 03.
    ("acta", float),  # 04.  (tcp orientation (3): euler angles a,b,c in
    #       absolute world coordinates)
    ("actb", float),  # 05.
    ("actc", float),  # 06.
    ("actg", int),  # 07.  (gripper_action (1): binary close=-1, open=1)
    ("relx", float),  # 08.  (tcp position (3): x,y,z in relative world coordinates
    #       normalized and clipped to (-1, 1) with scaling factor 50)
    ("rely", float),  # 09.
    ("relz", float),  # 10.
    ("rela", float),  # 11.  (tcp orientation (3): euler angles a,b,c in relative world
    #       coordinates normalized and clipped to (-1, 1) with scaling
    #       factor 20)
    ("relb", float),  # 12.
    ("relc", float),  # 13.
    ("relg", int),  # 14.  (gripper_action (1): binary close=-1, open=1)
    ("tcpx", float),  # 15.  (tcp position (3): x,y,z in world coordinates)
    ("tcpy", float),  # 16.
    ("tcpz", float),  # 17.
    ("tcpa", float),  # 18.  (tcp orientation (3): euler angles a,b,c in world
    #       coordinates)
    ("tcpb", float),  # 19.
    ("tcpc", float),  # 20.
    ("tcpg", float),  # 21.  (gripper opening width (1): in meters)
    ("arm1", float),  # 22.  (arm_joint_states (7): in rad)
    ("arm2", float),  # 23.
    ("arm3", float),  # 24.
    ("arm4", float),  # 25.
    ("arm5", float),  # 26.
    ("arm6", float),  # 27.
    ("arm7", float),  # 28.
    ("armg", int),  # 29.  (gripper_action (1): binary close = -1, open = 1)
    ("slider", float),  # 30.  (1): joint state: range=[-0.002359:0.306696]
    ("drawer", float),  # 31.  (1): joint state: range=[-0.002028:0.221432]
    ("button", float),  # 32.  (1): joint state: range=[-0.000935:0.033721]
    ("switch", float),  # 33.  (1): joint state: range=[-0.004783:0.091777]
    ("lightbulb", int),  # 34.  (1): on=1, off=0
    ("greenlight", int),  # 35.  (1): on=1, off=0
    ("redx", float),  # 36.  (red block (6): (x, y, z, euler_x, euler_y, euler_z)
    ("redy", float),  # 37.
    ("redz", float),  # 38.
    ("reda", float),  # 39.
    ("redb", float),  # 40.
    ("redc", float),  # 41.
    ("bluex", float),  # 42.  (blue block (6): (x, y, z, euler_x, euler_y, euler_z)
    ("bluey", float),  # 43.
    ("bluez", float),  # 44.
    ("bluea", float),  # 45.
    ("blueb", float),  # 46.
    ("bluec", float),  # 47.
    ("pinkx", float),  # 48.  (pink block (6): (x, y, z, euler_x, euler_y, euler_z)
    ("pinky", float),  # 49.
    ("pinkz", float),  # 50.
    ("pinka", float),  # 51.
    ("pinkb", float),  # 52.
    ("pinkc", float),  # 53.
]
dtype_cont = [
    ("idnum", int),  # 00.00
    ("slider.x", float),  # 01.54
    ("slider.y", float),  # 02.55
    ("slider.z", float),  # 03.56
    ("drawer.x", float),  # 04.57
    ("drawer.y", float),  # 05.58
    ("drawer.z", float),  # 06.59
    ("button.x", float),  # 07.60
    ("button.y", float),  # 08.61
    ("button.z", float),  # 09.62
    ("switch.x", float),  # 10.63
    ("switch.y", float),  # 11.64
    ("switch.z", float),  # 12.65
]

dtype_tact = [
    ("idnum", int),  # 00.00 idnum
    ("tact1d", float),  # 01.66 depth_tactile1
    ("tact2d", float),  # 02.67 depth_tactile2
    ("tact1r", float),  # 03.68 rgb_tactile1_r
    ("tact1g", float),  # 04.69 rgb_tactile1_g
    ("tact1b", float),  # 05.70 rgb_tactile1_b
    ("tact2r", float),  # 06.71 rgb_tactile2_r
    ("tact2g", float),  # 07.72 rgb_tactile2_g
    ("tact2b", float),  # 08.73 rgb_tactile2_b
]

dtype_lang = [("start", int), ("end", int), ("task", object), ("annot", object)]

act_range = range(1, 8)
rel_range = range(8, 15)
tcp_range = range(15, 22)
arm_range = range(22, 30)
scene_range = range(30, 54)
controller_range = range(54, 66)
tactile_range = range(66, 74)

int2task = [
    "close_drawer",
    "lift_blue_block_drawer",
    "lift_blue_block_slider",
    "lift_blue_block_table",
    "lift_pink_block_drawer",
    "lift_pink_block_slider",
    "lift_pink_block_table",
    "lift_red_block_drawer",
    "lift_red_block_slider",
    "lift_red_block_table",
    "move_slider_left",
    "move_slider_right",
    "open_drawer",
    "place_in_drawer",
    "place_in_slider",
    "push_blue_block_left",
    "push_blue_block_right",
    "push_into_drawer",
    "push_pink_block_left",
    "push_pink_block_right",
    "push_red_block_left",
    "push_red_block_right",
    "rotate_blue_block_left",
    "rotate_blue_block_right",
    "rotate_pink_block_left",
    "rotate_pink_block_right",
    "rotate_red_block_left",
    "rotate_red_block_right",
    "stack_block",
    "turn_off_led",
    "turn_off_lightbulb",
    "turn_on_led",
    "turn_on_lightbulb",
    "unstack_block",
]

fieldnames = [
    "actx",
    "acty",
    "actz",
    "acta",
    "actb",
    "actc",
    "actg",
    "relx",
    "rely",
    "relz",
    "rela",
    "relb",
    "relc",
    "relg",
    "tcpx",
    "tcpy",
    "tcpz",
    "tcpa",
    "tcpb",
    "tcpc",
    "tcpg",
    "arm1",
    "arm2",
    "arm3",
    "arm4",
    "arm5",
    "arm6",
    "arm7",
    "armg",
    "slider",
    "drawer",
    "button",
    "switch",
    "lightbulb",
    "greenlight",
    "redx",
    "redy",
    "redz",
    "reda",
    "redb",
    "redc",
    "bluex",
    "bluey",
    "bluez",
    "bluea",
    "blueb",
    "bluec",
    "pinkx",
    "pinky",
    "pinkz",
    "pinka",
    "pinkb",
    "pinkc",
    "slider.x",
    "slider.y",
    "slider.z",
    "drawer.x",
    "drawer.y",
    "drawer.z",
    "button.x",
    "button.y",
    "button.z",
    "switch.x",
    "switch.y",
    "switch.z",
    "tact1d",
    "tact2d",
    "tact1r",
    "tact1g",
    "tact1b",
    "tact2r",
    "tact2g",
    "tact2b",
    "slider_diff",
    "drawer_diff",
    "button_diff",
    "switch_diff",
    "lightbulb_diff",
    "greenlight_diff",
    "redx_diff",
    "redy_diff",
    "redz_diff",
    "reda_diff",
    "redb_diff",
    "redc_diff",
    "bluex_diff",
    "bluey_diff",
    "bluez_diff",
    "bluea_diff",
    "blueb_diff",
    "bluec_diff",
    "pinkx_diff",
    "pinky_diff",
    "pinkz_diff",
    "pinka_diff",
    "pinkb_diff",
    "pinkc_diff",
]
