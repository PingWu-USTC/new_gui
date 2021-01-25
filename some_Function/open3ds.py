# 打开 grid 文件
import os
import nanonispy as nap
import numpy as np
import struct


def openup(file_path):
    if os.path.splitext(file_path)[-1][1:] == '3ds':
        example = nap.read.Grid(file_path)
        _3ds_read(example, file_path)
    elif os.path.splitext(file_path)[-1][1:] == 'sxm':
        example1 = nap.read.Scan(file_path)
        _sxm_read(example1, file_path)
    elif os.path.splitext(file_path)[-1][1:] == 'dat':
        example2 = nap.read.Spec(file_path)
        Data = _dat_read(example2,file_path)
        return Data


def _3ds_read(example, file_path):
    paths, files = os.path.split(file_path)
    name = os.path.splitext(files)[0]
    rs = os.path.splitext(files)[-1][1:]
    New_path = paths + '/rawdata' + '_' + name + '_' + rs
    # lists = list(example.signals.keys())

    if not os.path.exists(New_path):
        os.makedirs(New_path)
    position_xy = example.header.get('pos_xy')
    dim = example.header.get('dim_px')
    nx = dim[0]
    ny = dim[1]
    #current = example.header.get('current')
    #bias = example.header.get('bias')
    size_xy = example.header.get('size_xy')
    point_x = position_xy[0]
    point_y = position_xy[1]
    lx = size_xy[0]
    ly = size_xy[1]
    x = np.linspace(point_x-lx/2,point_x+lx/2,nx)
    y = np.linspace(point_y-ly/2,point_y+ly/2,ny)
    sweep_signal = example.signals.get('sweep_signal')
    # print(x,y)

    for i in example.signals.keys():
        x_write=x
        y_write=y
        if i == 'params':
            pass
        else:
            result = i.rfind('/') != -1
            if result:
                i_new = i.replace('/', '_')
            else:
                i_new = i
            file = open(New_path + '/' + i_new + '.bin', 'wb')


            # 这里我们需要定义binary file 的头文件
            data = example.signals.get(i)
            print(type(data))
            # 这里得到的数据是元组数据
            shapes = data.shape
            print(shapes)
            if len(shapes) == 3:

                data = np.transpose(data, (1, 0, 2))  # 对前两个维度的坐标进行转置
                if example.header.get("angle") != 180:
                    data = np.flip(data, axis=1)
                    y_write = np.flip(y_write)
                if example.header.get("angle") == 90:
                    data = np.rot90(data, -1, (0, 1))  # 参数-1代表顺时针旋转90，（0，1）表示将data前两个维度的坐标进行旋转
                    # data = np.flip(data, axis=1)
                    data = np.transpose(data, (1, 0, 2))



                write_three_dimensional_bin(file, data, sweep_signal, x, y_write)
            if len(shapes) == 2:
                data = data.transpose()
                if example.header.get("angle") != 180:
                    data = np.flip(data, 1)
                    y_write = np.flip(y_write)
                if example.header.get("angle") == 90:
                    data = np.rot90(data, -1)
                    data = data.transpose()

                write_two_dimensional_bin(file, data, x, y_write)
            if len(shapes) == 1:
                write_one_dimensional_bin(file,data)
    return New_path

def _dat_read(example,filepath):
    Data = {}
    Data['BiasList']= example.signals.get('Bias calc (V)')
    Data['Current']= example.signals.get('Current (A)')
    Data['didv'] = example.signals.get('LI Demod 1 X (A)')
    return Data
def _sxm_read(example, file_path):
    paths, files = os.path.split(file_path)
    name = os.path.splitext(files)[0]
    rs = os.path.splitext(files)[-1][1:]
    New_path = paths + '/rawdata' + '_' + name + '_' + rs
    if not os.path.exists(New_path):
        os.makedirs(New_path)
    position_xy = example.header.get('scan_offset')
    dim = example.header.get('scan_pixels')
    nx = dim[0]
    ny = dim[1]
    size_xy = example.header.get('scan_range')

    point_x = position_xy[0]
    point_y = position_xy[1]
    lx = size_xy[0]
    ly = size_xy[1]
    x = np.linspace(point_x - lx / 2, point_x + lx / 2, nx)
    y = np.linspace(point_y - ly / 2, point_y + ly / 2, ny)
    for i in example.signals.keys():
        if i == 'params':
            pass
        else:
            result = i.rfind('/') != -1
            if result:
                i_new = i.replace('/', '_')
            else:
                i_new = i
            data = example.signals.get(i)
            if isinstance(data, dict):
                for keys in data.keys():
                    file = open(New_path + '/' + i_new + '_' + keys + '.bin', 'wb')
                    datas = data[keys]
                    # data = data.transpose()
                    # if example.header.get("angle") != 180:
                    #     data=np.flip(data,1)
                    #     y = np.flip(y)


                    write_two_dimensional_bin(file,datas,x,y)
            else:
                file = open(New_path, '/' + i_new + '.bin', 'wb')
                datas = data[keys]

                write_two_dimensional_bin(file,datas,x,y)
    return New_path

def write_three_dimensional_bin(file,data,sweep_signals,x,y):

    len_x,len_y,len_z = np.shape(data)
    layername=[None]*len_z
    for j in range(len_z):
        if j<10:
            layername[j]="00000"+str(j)
        elif j in range(10,100):
            layername[j]="0000"+str(j)
        elif j in range(100,1000):
            layername[j]="000"+str(j)
        elif j in range(1000,10000):
            layername[j]="00"+str(j)
        else:
            layername[j]=""+str(j)
    datas = struct.pack('>i',len_x)
    file.write(datas)
    datas = struct.pack('>i',len_y)
    file.write(datas)
    datas = struct.pack('>i',len_z)
    file.write(datas)
    z = sweep_signals
    for i in range(len_x):
        datas =struct.pack('>d',x[i])
        file.write(datas)
    for i in range(len_y):
        datas = struct.pack('>d',y[i])
        file.write(datas)
    for i in range(len_z):
        datas = struct.pack('>d',z[i])
        file.write(datas)
    for n in range(len_z):
        for i in range(len_x):
            for j in range(len_y):
                datas = struct.pack('>d',data[i][j][n])
                file.write(datas)

    for j in range(len_z):
        data = struct.pack(">i", len(layername[j]))
        file.write(data)

        for i in range(len(layername[j])):
            data =struct.pack("H",int(layername[j][i]))
            name="\x00"+layername[j][i]
            # data = bytes(name, "UTF-8")
            # data = bytes("0"+layername[j][i], "UTF-8")


            # print(data)
            file.write(data)



        # data = bytes(layername[j], "UTF-8")

        # file.write(layername[j])


    file.close()

def write_two_dimensional_bin(file,data,x,y):



    len_x,len_y = np.shape(data)

    datas = struct.pack('>i',len_x)
    file.write(datas)
    datas = struct.pack('>i',len_y)
    file.write(datas)
    datas=struct.pack('>d',1.0)
    file.write(datas)
    datas=struct.pack('>d',1.0)
    file.write(datas)
    for i in range(len_x):
        datas = struct.pack('>d', x[i])
        file.write(datas)
    for i in range(len_y):
        datas = struct.pack('>d', y[i])
        file.write(datas)
    for i in range(len_x):
        for j in range(len_y):
            datas = struct.pack('>d', data[i][j])
            file.write(datas)
    file.close()


def write_one_dimensional_bin(file,data):
    len_x = len(data)
    datas = struct.pack('>i',len_x)
    file.write(datas)
    for i in range(len_x):
        datas = struct.pack('>d',data[i])
        file.write(datas)
    file.close()


def write_txt_file(filepath,ans,name):
    New_path = filepath
    import os
    if not os.path.exists(New_path):
        os.makedirs(New_path)
    if os.path.exists(New_path+'/'+name+'.txt'):
        os.remove(New_path+'/'+name+'.txt')
    file = open(New_path+'/'+name+'.txt','a')
    len_x,len_y = np.shape(ans)
    for i in range(len_y):
        file.write(str(i+1))
        file.write('\t')
    file.write('\n')
    for i in range(len_x):
        file.write(str(i+1))
        file.write('\t')
        for j in range(len_y):
            file.write(str(ans[i][j]))
            file.write('\t')
        file.write('\n')
    file.close()


def write_txt_without_series(filepath,ans,name):
    New_path = filepath
    import os
    if not os.path.exists(New_path):
        os.makedirs(New_path)
    if os.path.exists(New_path+'/'+name+'.txt'):
        os.remove(New_path+'/'+name+'.txt')
    file = open(New_path+'/'+name+'.txt','a')
    len_x,len_y = np.shape(ans)
    for i in range(len_x):
        for j in range(len_y):
            file.write(str(ans[i][j]))
            file.write('\t')
        file.write('\n')
    file.close()
#def write_txt_without_series()
