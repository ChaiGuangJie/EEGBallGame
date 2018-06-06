import socket
import struct,queue
from enum import Enum
import time
# remote = ('127.0.0.1',8888) #根据服务器设置
#
# s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#
# s.connect(remote) #todo 界面上的点提示网络是否正常

cmdType = ''

def logSave(msg):
    print(msg)

class headerPara(Enum):
    code_generalCtrl = 1
    req_Ver = 1
    req_closeUpConn = 2

    code_serverCtrl = 2
    req_startAcq = 1
    req_stopAcq = 2
    req_startImp = 3
    req_changeSetup = 4
    req_DCCorr = 5

    code_clientCtrl = 3
    req_EDFHeader = 1
    req_ASTSetupFile = 2
    req_startSendingData = 3
    req_stopSendingData = 4
    req_basicInfo = 5

    code_setupFile = 1
    req_neuASTFormat = 1

    code_info = 1
    req_verInfo = 1
    req_standardEDFHeader = 2

    code_eegData = 2
    req_16bitRawData = 1
    req_32bitRawData = 2

class acqMessage(object):
    headSize = 12
    def __init__(self,chId,code,request,bSize):
        self.chId = chId
        self.code = code
        self.request = request
        self.bSize = bSize
        #elf.body = ""
    def convertHeader(self):
        return struct.pack('!4sHHI',self.chId,int(self.code),int(self.request),int(self.bSize))

class acqHeader(acqMessage):
    def __init__(self,buffer):
        self.unpackData = struct.unpack('!4sHHI',buffer)
        logSave('saved header:')
        logSave(self.unpackData)
        acqMessage.__init__(self,self.unpackData[0],self.unpackData[1],self.unpackData[2],self.unpackData[3])


    def IsCtrlPacket(self):
        logSave('ctrl packet')
        if self.chId == b'CTRL':
            return True
        else:
            return False

    def IsDataPacket(self):
        logSave('data packet')
        if self.chId == b'DATA':
            return True
        else:
            return False



class scanTransport():
    def __init__(self,host,port):
        self.s = socket.socket(socket.AF_INET,socket.SOCK_STREAM)

        self.host = host
        self.port = port
        self.clientHost = socket.gethostbyname(socket.gethostname())
        self.clientPort = 4000
        self.transportActive = False
        self.otherSideDown = False
        self.blocksize = 4096
        self.maxRecvSize = 10240
        self.eegQueue = queue.Queue()  # todo 设置超时选项等

    def sendEx(self,header,bufferData):
        if not bufferData:
            header.bSize = 0
        bSize = header.bSize
        self.s.sendall(header.convertHeader())
        if bSize:
            #todo 客户端好像不能发信息？
            logSave('客户端发送消息内含data body')


    def sendCommand(self,command,subtype):
        if not self.s:
            return False
        acqMsg = acqMessage(b'CTRL',command,subtype,0)
        self.sendEx(acqMsg,None)
        return True

    # def sendCommandEx(self,cmdTypeTuple):
    #     return self.sendCommand(cmdTypeTuple[0],cmdTypeTuple[1])

    def disconnect(self):
        if(self.s):
            #self.sendCommand(headerPara.code_clientCtrl.value,headerPara.req_stopAcq.value)#会发送EDFfile
            self.sendCommand(headerPara.code_clientCtrl.value,headerPara.req_stopSendingData.value)
            self.sendCommand(headerPara.code_generalCtrl.value,headerPara.req_closeUpConn.value)
            #time.sleep(0.1)
            logSave('finish disconnect!')
        self.transportActive=False

    def start(self):
        try:
            #self.s.bind((self.clientHost,self.clientPort))#指定本机ip和端口 不然程序奔溃后重连需要scan重新开始记录
            self.s.connect((self.host,self.port))
        except Exception as e:
            logSave('连接出错')
            logSave(e)
            return False
        #todo initVars
        logSave('连接成功')
        # self.sendCommand(headerPara.code_clientCtrl.value,headerPara.req_EDFHeader.value)
        # logSave('发送命令成功')
        return True

    def stop(self):
        self.disconnect()
        return True

    def processCtrlMsg(self, header, bufferData):
        logSave('process ctrl msg!')
        if header.code == headerPara.code_generalCtrl.value:
            if header.request == headerPara.req_closeUpConn.value:
                self.transportActive = False
                self.otherSideDown = True
        elif header.code == headerPara.code_serverCtrl.value:
            if header.request == headerPara.req_startAcq.value:
                #todo initVars
                logSave('req_startAcq')
                pass
        else:
            logSave('other header:', header)
        logSave('finish process ctrl msg')
        return True

    def processDataMsg(self, header, bufferData):
        logSave('process data msg')
        if header.code == headerPara.code_info.value:
            if header.request == headerPara.req_EDFHeader.value:
                logSave('req_EDFHeader')
                pass
                # todo parase edf file
            logSave('dataInfo:')
            logSave(bufferData)
        elif header.code == headerPara.code_eegData.value:
            dataLen = 0
            if header.request == headerPara.req_16bitRawData.value:
                dataLen = int(header.bSize/2)
                # unpackData = struct.unpack('!'+ dataLen + 'h',bufferData)
                # print(unpackData[:100])
            elif header.request == headerPara.req_32bitRawData.value:
                dataLen = int(header.bSize/4)
                #todo 判断标志位stopRecvData 每次装Q之前先清空？
            # print(bufferData)
            # print(type(bufferData))
            print('dataLen:',dataLen)
            fmt = '!' + str(dataLen) + 'i'
            print(fmt)
            unpackData = struct.unpack(fmt,bufferData)
            self.eegQueue.put(unpackData)#unpack数据在什么时候组装成网络的输入 主要看哪个环节最不费时，且queue是否阻塞
            print(unpackData[:100])

        logSave('finish processDataMsg')
        return True

    def recvData(self):
        self.transportActive = True
        logSave('开始接收数据')
        while self.transportActive:#一次循环接收一个数据包(stream packet)
            try:
                bufferHeader = self.s.recv(acqMessage.headSize,socket.MSG_WAITALL)#,socket.MSG_WAITALL acqMessage.headSize
            except ConnectionResetError as e:
                logSave(e)
                break
            logSave('数据头部接收成功')
            print('bufferHeader:',bufferHeader)
            if bufferHeader:
                header = acqHeader(bufferHeader)
                logSave('header.bSize:')
                logSave(header.bSize)
                totalRecved = 0
                bufferData = []
                # print(self.s.recv(12))
                while totalRecved < header.bSize:
                    try:
                        print('subBuff recv begin')
                        #self.s.settimeout(1)
                        buffersize = header.bSize - totalRecved if header.bSize - totalRecved < self.maxRecvSize else self.maxRecvSize
                        print('buffersize:',buffersize)
                        subBuff = self.s.recv(buffersize)
                        if len(subBuff) == 0:
                            break
                        print('subBuff:',subBuff)
                        print('len(subBuff):',len(subBuff))
                    except Exception as e:
                        logSave(e)
                        break
                    bufferData.append(subBuff)
                    totalRecved +=  len(subBuff)
                    print('totalRecved:',totalRecved)
                    print('header.bSize:',header.bSize)

                if header.IsCtrlPacket():
                    self.processCtrlMsg(header, b''.join(bufferData))
                elif header.IsDataPacket():
                    self.processDataMsg(header, b''.join(bufferData))
            else:
                logSave('bufferHeader 为空')
                break

            self.sendCommand(headerPara.code_clientCtrl.value,headerPara.req_stopSendingData.value)
            self.transportActive = False
    def recvDataWithoutHeader(self):
        # while self.transportActive:
        #     block = self.s.recv(self.blocksize)
        blockHeader = self.s.recv(self.blocksize)
        print(blockHeader)
        print('len(blockHeader):',len(blockHeader))
        bh = acqHeader(blockHeader)
        print(bh)

    def selectCommand(self):
        cmdDict = {
            '1': [headerPara.code_clientCtrl.value, headerPara.req_EDFHeader.value],
            '2': [headerPara.code_clientCtrl.value, headerPara.req_ASTSetupFile.value],
            '3': [headerPara.code_clientCtrl.value, headerPara.req_startSendingData.value],
            '4': [headerPara.code_clientCtrl.value, headerPara.req_stopSendingData.value],
            '5': [headerPara.code_clientCtrl.value,headerPara.req_stopAcq.value]
            # 'q': [headerPara.code_generalCtrl.value, headerPara.req_closeUpConn.value]
        }
        loop = True
        while loop:
            command = input('please input command:')
            if command in cmdDict:
                self.sendCommand(cmdDict[command][0],cmdDict[command][1])
                threading.Thread(target=self.recvData).start()
            elif command == 'q':
                    self.disconnect()
                    #self.sendCommand(headerPara.code_generalCtrl.value, headerPara.req_closeUpConn.value)
                    loop = False
            else:
                print('无效命令',command[0],command[-1])
                self.sendCommand(command[0], command[-1])
        logSave('退出循环')



if __name__ == '__main__':
    import threading
    try:

        trans = scanTransport('159.226.19.197',4000)
        trans.start()
        recv = threading.Thread(target=trans.recvData,name='recvData')
        send = threading.Thread(target=trans.selectCommand,name='sendCmd')

        send.start()
        #recv.start()
        send.join()
        #recv.join()
        print('end try')
    finally:
        if trans.transportActive:
            trans.disconnect()
        print('finally')
    # selectCommand()