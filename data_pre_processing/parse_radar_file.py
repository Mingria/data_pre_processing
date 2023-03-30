import struct
import numpy as np
#import cartopy.crs as ccrs
import numpy.ma as ma

class diamond131:
    ZoneName = None
    DataName = None
    Flag = None
    Version = None
    Year = None
    Month = None
    Day = None
    Hour = None
    Minute = None
    Interval = None
    XNumGrids = None
    YNumGrids = None
    ZNumGrids = None
    RadarCount = None
    StartLon = None
    StartLat = None
    CenterLon = None
    CenterLat = None
    XReso = None
    YReso = None
    ZHighGrids = None
    RadarStationName = None
    RadarLongtitude = None
    RadarLatitude = None
    RadarAltitude = None
    MosaicFlag = None
    Reserved = None
    ObserverValue = None

    def ReadFile(self,filePath,ref_flag = True):
        binfile= open(filePath,'rb')
        self.ZoneName = struct.unpack('12c',binfile.read(12));
        self.DataName = struct.unpack('38c',binfile.read(38));
        self.Flag = struct.unpack('8c',binfile.read(8));
        self.Version = struct.unpack('8c',binfile.read(8));
        self.Year = struct.unpack('h',binfile.read(2))[0];
        self.Month = struct.unpack('h',binfile.read(2))[0];
        self.Day = struct.unpack('h',binfile.read(2))[0];
        self.Hour = struct.unpack('h',binfile.read(2))[0];
        self.Minute = struct.unpack('h',binfile.read(2))[0];
        self.Interval = struct.unpack('h',binfile.read(2))[0];
        self.XNumGrids = struct.unpack('h',binfile.read(2))[0];
        self.YNumGrids = struct.unpack('h',binfile.read(2))[0];
        self.ZNumGrids= struct.unpack('h',binfile.read(2))[0];
        self.RadarCount = struct.unpack('i',binfile.read(4))[0];
        self.StartLon = round(struct.unpack('f',binfile.read(4))[0],2) ;    #经度
        self.StartLat = round(struct.unpack('f',binfile.read(4))[0],2);     #纬度
        self.CenterLon = round(struct.unpack('f',binfile.read(4))[0],2);    #中心经度
        self.CenterLat = round(struct.unpack('f',binfile.read(4))[0],2);    #中心维度
        self.XReso = round(struct.unpack('f',binfile.read(4))[0],2);        #x分辨率
        self.YReso = round(struct.unpack('f',binfile.read(4))[0],2);        #y分辨率
        self.ZHighGrids = struct.unpack('40f',binfile.read(160));
        self.RadarStationName = struct.unpack('320c',binfile.read(320));
        self.RadarLongtitude = struct.unpack('20f',binfile.read(80));
        self.RadarLatitude = struct.unpack('20f',binfile.read(80));
        self.RadarAltitude = struct.unpack('20f',binfile.read(80));
        self.MosaicFlag = struct.unpack('20?',binfile.read(20));
        self.Reserved = struct.unpack('172c',binfile.read(172));
        observerDataCount = self.XNumGrids*self.YNumGrids*self.ZNumGrids
        if(ref_flag):
            observerValue = struct.unpack(str(observerDataCount) + 'B', binfile.read(observerDataCount));
            self.ObserverValue = np.array(observerValue).reshape(self.ZNumGrids, self.YNumGrids, self.XNumGrids)
            self.ObserverValue = (self.ObserverValue.astype(np.float32) - 66)/2
            self.ObserverValue[self.ObserverValue <= 0] = 0


        else:
            observerValue = struct.unpack(str(observerDataCount) + 'h', binfile.read(observerDataCount * 2));
            self.ObserverValue = np.array(observerValue).reshape(self.ZNumGrids, self.YNumGrids, self.XNumGrids)
            self.ObserverValue =  self.ObserverValue.astype(np.float32)/10

            self.ObserverValue[self.ObserverValue == -33] = 0




    def writefile(self,filePath,ref_flag = True):
        with open(filePath, 'wb')as fp:
            writebin = struct.pack('12c',*self.ZoneName)
            fp.write(writebin)
            writebin = struct.pack('38s', self.DataName.encode('gbk'))
            fp.write(writebin)
            writebin = struct.pack('8c', *self.Flag)
            fp.write(writebin)
            writebin = struct.pack('8c', *self.Version)
            fp.write(writebin)
            writebin = struct.pack('h', self.Year)
            fp.write(writebin)
            writebin = struct.pack('h', self.Month)
            fp.write(writebin)
            writebin = struct.pack('h', self.Day)
            fp.write(writebin)
            writebin = struct.pack('h', self.Hour)
            fp.write(writebin)
            writebin = struct.pack('h', self.Minute)
            fp.write(writebin)
            writebin = struct.pack('h', self.Interval)
            fp.write(writebin)
            writebin = struct.pack('h', self.XNumGrids)
            fp.write(writebin)
            writebin = struct.pack('h', self.YNumGrids)
            fp.write(writebin)
            writebin = struct.pack('h', self.ZNumGrids)
            fp.write(writebin)
            writebin = struct.pack('i', self.RadarCount)
            fp.write(writebin)
            writebin = struct.pack('f', self.StartLon)
            fp.write(writebin)
            writebin = struct.pack('f', self.StartLat)
            fp.write(writebin)
            writebin = struct.pack('f', self.CenterLon)
            fp.write(writebin)
            writebin = struct.pack('f', self.CenterLat)
            fp.write(writebin)
            writebin = struct.pack('f', self.XReso)
            fp.write(writebin)
            writebin = struct.pack('f', self.YReso)
            fp.write(writebin)
            writebin = struct.pack('40f', *self.ZHighGrids)
            fp.write(writebin)
            writebin = struct.pack('320c', *self.RadarStationName)
            fp.write(writebin)
            writebin = struct.pack('20f', *self.RadarLongtitude)
            fp.write(writebin)
            writebin = struct.pack('20f', *self.RadarLatitude)
            fp.write(writebin)
            writebin = struct.pack('20f', *self.RadarAltitude)
            fp.write(writebin)
            writebin = struct.pack('20?', *self.MosaicFlag)
            fp.write(writebin)
            writebin = struct.pack('172c', *self.Reserved)
            fp.write(writebin)
            obsvalue = self.ObserverValue.reshape(self.ZNumGrids * self.YNumGrids * self.XNumGrids)
            if (ref_flag):
                #observerValue = struct.unpack(str(observerDataCount) + 'B', binfile.read(observerDataCount));
                #self.ObserverValue = np.array(observerValue).reshape(self.ZNumGrids, self.YNumGrids, self.XNumGrids)
                #self.ObserverValue = (self.ObserverValue.astype(np.float32) - 66) / 2
                #self.ObserverValue[self.ObserverValue == -33] = 0
                obsvalue = obsvalue * 2 + 66
                obsvalue[obsvalue == 66] = 0
                obsvalue = obsvalue.astype(np.char)
                writebin = struct.pack('%dc'%self.ZNumGrids*self.YNumGrids*self.XNumGrids, *obsvalue)
                fp.write(writebin)
            else:
                obsvalue = obsvalue*10
                obsvalue = obsvalue.astype(np.int16) #赋值操作后a的数据类型变化
                writebin = struct.pack('%dh' % self.ZNumGrids * self.YNumGrids * self.XNumGrids,*obsvalue)
                fp.write(writebin)


    def GetLonLatArray(self):
        endLon = self.StartLon + 2*(self.CenterLon - self.StartLon)
        endLat = self.StartLat + 2*(self.CenterLat - self.StartLat)
        xflag = (1 if(self.StartLon<self.CenterLon) else -1)
        yflag =  (1 if(self.StartLat<self.CenterLat) else -1)
        b_lon = (self.StartLon if(self.StartLon<self.CenterLon) else endLon)
        e_lon = (endLon if(self.StartLon<self.CenterLon) else self.StartLon)
        b_lat =  (self.StartLat if(self.StartLat<self.CenterLat) else endLat)
        e_lat =  (endLat if(self.StartLat<self.CenterLat) else self.StartLat)
        print(self.StartLon,self.StartLat,endLon,endLat,self.CenterLon,self.CenterLat,self.YNumGrids)
        return np.mgrid[ self.StartLat:endLat:self.YReso*yflag,self.StartLon:endLon:self.XReso*xflag],b_lon,e_lon,b_lat,e_lat


