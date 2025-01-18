import matplotlib.font_manager

# 获取系统中所有可用字体信息
font_list = matplotlib.font_manager.findSystemFonts(fontpaths=None, fontext='ttf')

# 扗印所有可用字体名称
for font_path in font_list:
    font_name = matplotlib.font_manager.FontProperties(fname=font_path).get_name()
    print(font_name)
