import numpy as np
from Colormap import *
from matplotlib.colors import  LinearSegmentedColormap
class unit_for_all:

    def __init__(self,x,label ='length'):
        """
        :param x:需要处理的单位类型
        :param label: 我们这里的标签主要是针对数据类型来的，存在 length(m) Current(A) Bias(V) DI/DV(S)，主要是这里的单位转换
        """
        self.label = label

        if self.label =='length':
            if np.abs(x)>=1:
                self.x = x
                self.units = 'm'
            elif np.abs(x)<1 and np.abs(x)>=10**-3:
                self.x = x*10**3
                self.units = 'mm'
            elif np.abs(x)< 10**-3 and np.abs(x)>=10**-6:
                self.x = x*10**-6
                self.units = '$\mu$m'
            elif np.abs(x)<10**-6 and np.abs(x)>=10**-9:
                self.x = x*10**9
                self.units = 'nm'
            elif np.abs(x)<10**-9 and np.abs(x)>10**-12:
                self.x = x*10**12
                self.units = 'fm'

class singlelayer:
    def __init__(self,layername,data,nx,ny,x,y,v,current,x_label,y_label):
        """
        :param layername: 层名
        :param data: 保存的二维数组
        :param nx: x 方向长度
        :param ny: y 方向长度
        :param x: x 方向每个pixel的index
        :param y:
        :param v: 所测电压
        :param current: 所测setpoint
        :param new_x x方向每个像素对应的长度
        :param new_y y方向每个像素对应的长度
        """
        self.layername = layername
        self.data = data
        self.nx = nx
        self.ny = ny
        self.x = x
        self.sizex =np.abs(x[0]-x[nx-1])
        self.y = y
        self.sizey = np.abs(y[0]-y[ny-1])
        self.bias = v
        self.current = current
        self.xlabel = x_label
        self.ylabel = y_label
        self.layername = layername
        self.x_ticket = unit_for_all(self.sizex,label =self.xlabel)
        self.y_ticket = unit_for_all(self.sizey,label =self.ylabel)
        self.layersize = 'Size'+'\t'+'('+'X'+':'+' '+str(round(self.x_ticket.x,1))+' '+self.x_ticket.units+';'+'Y'+':'+' '+str(round(self.y_ticket.x,1))+' '+self.y_ticket.units+')'+'\n'
        self.layerpixel = 'Pixel'+'\t'+'('+'X'+':'+' '+str(self.nx)+' '+'Y'+':'+' '+str(self.ny)+')'+'\n'
        self.Text = self.layername+'\n'+self.layersize+self.layerpixel
        self.new_x = np.linspace(0,self.y_ticket.x,nx)
        self.new_y = np.linspace(0,self.x_ticket.x,ny)

class draw_information_line_topo:
    def __init__(self,layer_data:singlelayer):
        """
        字典数据类型
        :param Line_and_Ticks{bottom/top/right/left:
                {Linewidth: float;
                 MajorTicks:
                            {style : in/out/none(str);
                             length : float;
                             thickness : float}
                 MinorTicks:
                            {style : in/out/none(str);
                             length : float;
                             thickness : float}
                 }
        :param title {xlabel:str;ylabel:str;font:str;size:int}
        :param Scale {direction:Vertical/horizontal:
                                {range: float~float
                                 majorticks:
                                    {style: By increment(str)
                                    value: int}
                                minorticks:
                                    {style: By Counts/By Count
                                    value: int}
                                }
                       }
        """
        self.layers = layer_data
        self.show_axis = 'off'
        self.scale = {'Vertical':{},'Horizontal':{}}
        self.scale['Vertical'] ={'Range':{'minus':np.min(self.layers.new_x),'maxus':np.max(self.layers.new_x)},'Majorticks':{'style':'By Increment','value':5},'Minorticks':{'style':'By Counts','value':10}}
        self.scale['Horizontal'] ={'Range':{'minus':np.min(self.layers.new_y),'maxus':np.max(self.layers.new_y)},'Majorticks':{'style':'By Increment','value':5},'Minorticks':{'style':'By Counts','value':10}}
        self.title={'xlabel':self.layers.xlabel+' '+'('+self.layers.x_ticket.units+')','ylabel':self.layers.ylabel+' '+'('+self.layers.y_ticket.units+')','font':'Times New Roman','size':22}
        self.line_and_ticks = {'Bottom':{},'Top':{},'Right':{},'Left':{}}
        self.line_and_ticks['Bottom'] = {'Linewidth':2,'MajorTicks':{'style':'in','length':6,'thickness':2,'fontsize':12},'MinorTicks':{'style':'in','length':3,'thickness':2}}
        self.line_and_ticks['Top'] = {'Linewidth':2,'MajorTicks':{'style':'in','length':6,'thickness':2,},'MinorTicks':{'style':'in','length':3,'thickness':2}}
        self.line_and_ticks['Right'] = {'Linewidth':2,'MajorTicks':{'style':'in','length':6,'thickness':2,},'MinorTicks':{'style':'in','length':3,'thickness':2}}
        self.line_and_ticks['Left'] = {'Linewidth':2,'MajorTicks':{'style':'in','length':6,'thickness':2,'fontsize':12},'MinorTicks':{'style':'in','length':3,'thickness':2}}
        """
        这些量到底对什么比较重要
        """



    def append_scale_information(self,V_or_H,Range_or_Majorticks_or_Minorticks,former_keys,text_information):
        self.scale[V_or_H][Range_or_Majorticks_or_Minorticks][former_keys] = text_information


    def append_title_information(self,xlabel_or_ylabel_or_font_size,text_information):
        self.title[xlabel_or_ylabel_or_font_size] = text_information

    def append_line_and_tickes_information(self,b_or_t_r_l,Line_majortickes_minortickes,*args):
        if len(args) ==2:
            self.line_and_ticks[b_or_t_r_l][Line_majortickes_minortickes][args[0]] = args[1]
        elif len(args)==1:
            self.line_and_ticks[b_or_t_r_l][Line_majortickes_minortickes] = args[0]
        else:
            print("tickes_index_error")
    def append_show_axis(self,off_or_on):
        self.show_axis = off_or_on
    def detect_empty(self):
        index_empty =True
        print(self.scale)
        self.send_information = "Now There are empty:\n"
        if bool(self.line_and_ticks):
           for i in self.line_and_ticks.keys():
                for j in self.line_and_ticks[i].keys():
                    if j == 'Linewidth':
                        if self.line_and_ticks[i][j] == None:
                            index_empty = False
                            self.send_information=self.send_information +i+j+'is'+'empty'+'\n'
                    else:
                        for k in self.line_and_ticks[i][j].keys():
                            if self.line_and_ticks[i][j][k] == None:
                                index_empty =False
                                self.send_information = self.send_information + i +' '+j+' '+k+' '+'is' +' '+'empty' + '\n'
        else:
            index_empty = False
        if bool(self.scale):
            for i in self.scale.keys():
                for j in self.scale[i].keys():
                    for m in self.scale[i][j].keys():
                        if self.scale[i][j][m] == None:
                            index_empty = False
                            self.send_information = self.send_information + i +' '+j+' '+m+' '+'is' +' '+'empty' + '\n'
        else:
            index_empty = False
        if bool(self.title):
            for i in self.scale.keys():
                if self.scale[i] == None:
                    index_empty = False
                    self.send_information = self.send_information+self.send_information + i +' '+'is'+' '+'empty'+'\n'
        else:
            index_empty = False
        if not self.show_axis:
            index_empty = False
            self.send_information = self.send_information +'axis_on_or_off'+' '+'is'+'empty'+'\n'
        if index_empty:
            return index_empty, 'noMessage'
        else:
            return index_empty,self.send_information




class data_for_single_gui:
    def __init__(self,layername=None,layers=None,color_count=0,vmins=0,vmax=100):
        """
        :param cmaplist: colormap
        :param fftcount: 是否处于fft状态
        :param contrast ratio: 图像对比度调节
        :param draw_tools_index: 图片绘制参数设定
        这个类的设计目的就是为了对目前显示图形的整体的把握，包括需要显示的数据，色彩设定？图像细节参数等等。
        预留接口问题？如何在后续功能扩展中以最小的代价，实现新的功能的增加
        在这里我可以设想一下，如果以后要设定这种坐标轴并不是非线性的情况
        基于此，我设计的时候，首先我认为：无论是fft/topo/cut/laplace都是等价的，也就是layer的生成过程应该是在
        data_for_single_gui类的外部的。但是我认为在对于色彩、对比度、图形指数等与数据应该是同一层的，这样方便在后续过程中修改
        也就是对于图形本身的性质，都是在这个类上进行操作的。至于如何将数据进行fft变换，我想这不是该类应该有的功能。
        """
        self.color_count = color_count
        self.vmins = vmins
        self.vmaxs = vmax

        """最关键的初始化。必须要考虑一下"""
        self.cmap = 'RdBu'
        self.cmaplist = ['YlOrBr','YlOrBr_r','Blues','Blues_r','binary_r','binary',
                         'autumn','autumn_r','YlOrRd','YlOrRd_r','Reds','Reds',
                         'Purples','Purples_r','copper','copper_r','OrRd','OrRd_r','RdBu',
                         'RdBu_r','coolwarm','coolwarm_r']
        self.colorlist,self.len_of_colorlist,self.cnamelist= colorlistfunction()
        self.nbins=1000#制作色彩的时候，色彩的梯度

        self.current_layer = {}
        self.layer_in_gui = {}
        self.current_layer[layername] = layers
        self.layer_in_gui[layername] = layers
        self.draw_index = draw_information_line_topo(layers)#初始化？必须引入数据

        #self.layer_in_gui = {}
        #self.current_layer = {}
        self.filepath = r'F:\CsV3Sb5'


    def change_filepath(self,filepaths):
        self.filepath = filepaths

    def change_color(self,count):
        """
        :param count:increase or decrease colorcount
        :return:cmap
        """
        self.color_count = self.color_count+count
        time = int(self.color_count%(len(self.cmaplist)+self.len_of_colorlist))
        if time<self.len_of_colorlist:
            self.cmap = LinearSegmentedColormap.from_list(self.cnamelist[time],self.colorlist[self.cnamelist[time]],N =self.nbins)
        else:
            self.cmap = self.cmaplist[time-self.len_of_colorlist]
        return self.cmap


    def change_contrast_ratio(self,vmin,vmax):
        """
        :param vmin:
        :param vmax:
        :return:
        重新设计F11功能键的属性(对比度更改)
        """
        self.vmins = vmin
        self.vmaxs = vmax

    def replace_layer(self,layer:singlelayer):
        if self.current_layer:
            self.current_layer.pop(list(self.current_layer.keys())[0])
        else:
            pass
        self.current_layer[layer.layername] = layer
        self.append_layer(layer.layername,layer)


    def return_current_layer(self):
        return list(self.current_layer.keys())[0],list(self.current_layer.values())[0]


    def return_current_layer_value(self):
        return list(self.current_layer.values())[0]


    def return_current_layer_name(self):
        return list(self.current_layer.keys())[0]
    """
    添加数据层
    """

    def append_layer(self,layername,layer:singlelayer):
        self.layer_in_gui[layername] = layer
    """ 
    删除数据层
    """
    def delete_layer(self,layername):
        del self.layer_in_gui[layername]


    def delete_all_layer(self):
        self.layer_in_gui.clear()
    """
    返回目前已有的数据层层名
    """

    def return_all_layers_name(self):
        return list(self.layer_in_gui.keys())


    def refresh_draw_index(self,draw_index:draw_information_line_topo):
        self.draw_index = draw_index


    def return_all_draw_index(self):
        return self.cmap,list(self.current_layer.keys())[0],list(self.current_layer.values())[0],self.draw_index,self.vmins,self.vmaxs

    def return_draw_index(self):
        return self.draw_index


"""
为每个GUI创建这个一个类似于字典一样的数据包
"""

"""
a = {}
if a:
    a.pop(list(a.keys())[0])
else:
    pass
a['ping'] = 'pings'

print(list(a.keys()))
print(list(a.values())[0])
print(a[list(a.keys())[0]])
"""


class multilayer:
    def __init__(self,data,nx,ny,x,y,v):
        self.data=data
        self.nx=nx
        self.ny=ny
        self.x=x
        self.sizex=abs(x[0]-x[nx-1])
        self.y=y
        self.sizey=abs(y[0]-y[ny-1])
        self.biaslist=v
        self.nlayers=len(self.biaslist)
        self.fftdata = np.zeros((np.shape(self.data)),dtype=complex)
        self.fftmodule = np.zeros((np.shape(self.data)))
        for i in range(self.nlayers):
            self.fftdata[i] = np.fft.fft2(self.data[i])
            self.fftmodule[i] = np.log10(np.abs(np.fft.fftshift(self.fftdata[i])))



