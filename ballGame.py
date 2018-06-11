import pygame
import EEGClient
import os
import threading
import random
import time
from tkinter.filedialog import asksaveasfilename, askopenfilename
#os.environ['SDL_VIDEO_WINDOW_POS'] = str(position[0]) + "," + str(position[1])
os.environ['SDL_VIDEO_CENTERED'] = '1'  # 窗口中央显示

#####################################
TT = 3  # target time limit (s)
BT = 1  # blank time limit  (s)
WT = 0  # 0.1  #watch ball time (s)
SRD = 200  # stride scale
########################################
A = 200  # 比SRD略大即可

########################################


class Ball(pygame.sprite.Sprite):
    image = pygame.image.load("ball.png")
    image.set_colorkey((0, 0, 0))
    grayImage = pygame.image.load("ball.png")
    grayImage.set_colorkey((0, 0, 0))
    #image = image.convert_alpha()
    #centerPos = (SCREENWIDTH/2,SCREENHEIGHT/2)

    def __init__(self, startpos):
        # 获取系统参数如屏幕
        pygame.sprite.Sprite.__init__(self, self.groups)
        self.pos = startpos
        self.image = Ball.grayImage.convert_alpha()
        self.rect = self.image.get_rect()
        self.rect.center = self.pos
        self._lastTime = 0

    def update(self, direction, stride=1):
        '''
        direction:-1到1之间的参数
        stride:小球一次运动的步长
        '''
        self.rect.centerx += direction * stride
        print('update stride:', direction * stride)
        currentTime = time.time()
        timeInterval = currentTime - self._lastTime
        self._lastTime = currentTime
        print("time interval:",timeInterval)


    def retart(self):
        self.rect.center = self.pos


class TargetArea(pygame.sprite.Sprite):
    areaWidth = 200
    areaHeight = 600
    defaultColor = (200, 200, 200)
    targetColor = (0, 150, 0)

    def __init__(self, left, top, width=areaWidth, height=areaHeight):
        pygame.sprite.Sprite.__init__(self, self.groups)
        self.left = left
        self.top = top
        self.width = width
        self.height = height
        self.image = pygame.Surface((width, height))
        self.image.fill(TargetArea.defaultColor)
        self.rect = self.image.get_rect()
        self.updateColor = TargetArea.defaultColor
        self.rect.center = (width / 2 + left, height / 2 + top)

    def update(self, target):
        if target:  # 边框（区域）变为亮色
            self.updateColor = TargetArea.targetColor
        else:  # 灰色
            self.updateColor = TargetArea.defaultColor
        #pygame.draw.rect(self.image, self.updateColor, [self.left, self.top, self.width, self.height])
        self.image.fill(self.updateColor)


class ballGame():
    def __init__(self):
        pygame.init()
        disInfo = pygame.display.Info()
        print(disInfo)
        self.SCREENWIDTH = disInfo.current_w
        self.SCREENHEIGHT = disInfo.current_h
        self.BACKGROUNDCOLOR = (255, 255, 255)
        self.INTERVAL = 500
        self.CENTER_X = self.SCREENWIDTH / 2
        self.CENTER_Y = self.SCREENHEIGHT / 2
        self.screen = pygame.display.set_mode(
            (self.SCREENWIDTH, self.SCREENHEIGHT), pygame.RESIZABLE)  # pygame.FULLSCREEN
        self.background = pygame.Surface(self.screen.get_size())

        self.clock = pygame.time.Clock()  # create pygame clock object
        self.mainloop = True
        self.FPS = 60  # desired max. framerate in frames per second.
        self.pause = True  # todo 实验时开启

        self.ballgroup = pygame.sprite.Group()
        self.allgroup = pygame.sprite.Group()
        Ball.groups = self.ballgroup, self.allgroup  # 这么设计真的好吗
        TargetArea.groups = self.allgroup

        self.ball = Ball((self.CENTER_X, self.CENTER_Y))
        self.normalBall = False  # 灰色 有颜色为True

        self.leftBoundary = self.CENTER_X - self.INTERVAL - TargetArea.areaWidth / 2
        self.rightBoundary = self.CENTER_X + self.INTERVAL - TargetArea.areaWidth / 2
        self.topBoundary = self.CENTER_Y - TargetArea.areaHeight / 2
        self.leftArea = TargetArea(self.leftBoundary, self.topBoundary)
        self.rightArea = TargetArea(self.rightBoundary, self.topBoundary)

        self.strideLock = threading.Lock()
        self.stopRecvLock = threading.Lock()
        self.labelLock = threading.Lock()
        self.keepOn = False
        self.label = 0
        self.stride = 0
        self.returnStride = False
        self.stopRecvData = True

        self.T = 1.0
        self.vt = 0
        self.A = 0
        self.offset = 0
        self.continued = 0
        #self.trans = EEGClient.scanTransport(remoteHost, remotePort)
        self.state = "暂停"
        self.trainState = ""

    def write(self, msg="ball game"):
        myfont = pygame.font.SysFont("微软雅黑", 32)
        mytext = myfont.render(msg, True, (0, 0, 0))
        mytext = mytext.convert_alpha()
        return mytext

    def initGameWindow(self):
        self.background.fill(self.BACKGROUNDCOLOR)  # fill white
        self.background.blit(self.write('+'), (self.CENTER_X, self.CENTER_Y))
        # background.blit(write("Press ESC to quit"),(300,10))
        self.background = self.background.convert()  # jpg can not have transparency
        # blit background on screen (overwriting all)
        self.screen.blit(self.background, (0, 0))
        #self.leftArea.update(1)

    def setBlank(self):
        self.leftArea.update(False)
        self.rightArea.update(False)
        # print('setBlank')

    def setTarget(self):
        r = random.randint(0, 1)

        self.setLabel(-1 if r == 0 else 1)

        self.rightArea.update(r)
        self.leftArea.update(1 - r)

    def setLabel(self, label):
        # self.labelLock.acquire()
        self.label = label
        # print('setLabel',label)
        # self.labelLock.release()

    def getLabel(self):
        # self.labelLock.acquire()
        label = self.label
        # self.labelLock.release()
        return label

    def getStopRecv(self):
        # self.stopRecvLock.acquire()
        recvData = self.stopRecvData
        # self.stopRecvLock.release()
        return recvData

    def stopRecv(self):
        self.stopRecvLock.acquire()
        self.stopRecvData = True
        self.stopRecvLock.release()

    def startRecv(self):
        self.stopRecvLock.acquire()
        self.stopRecvData = False
        self.stopRecvLock.release()

    def setStride(self, stride):
        self.strideLock.acquire()
        self.stride = stride
        self.returnStride = True
        #print('setStride')
        self.strideLock.release()

    def clearStride(self):
        self.strideLock.acquire()
        self.stride = 0
        self.returnStride = False
        self.strideLock.release()

    def getStride(self):
        # self.strideLock.acquire()
        stride = self.stride
        # self.strideLock.release()
        return stride

    def getOffsetDis(self, dt):
        print('self.vt:', self.vt)
        print('self.A:', self.A)
        s = self.vt * dt
        print('s:', s)
        lastVt = self.vt
        self.vt = self.vt + self.A * dt

        if lastVt * self.vt < 0:
            self.vt = 0
            self.A = 0

        # if self.A * s >= 0: #异号 小球开始反向运动
        #     self.vt = 0
        #     self.A = 0
        return s

    def collisionDetect(self):
        if self.ball.rect.centerx > self.rightArea.rect.centerx or self.ball.rect.centerx < self.leftArea.rect.centerx:
            pass

    def run(self,saveModel,loadModel):
        self.initGameWindow()

        #continued = 0
        isTargetTime = False
        targetTime = 0
        blankTime = 0
        holdOnTime = 0

        while self.mainloop:
            # milliseconds passed since last frame
            milliseconds = self.clock.tick(self.FPS)
            seconds = milliseconds / 1000.0  # seconds passed since last frame
            self.continued += seconds
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.mainloop = False  # pygame window closed by user
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.mainloop = False  # user pressed ESC
                    elif event.key == pygame.K_LEFT:
                        self.setStride(-1)
                    elif event.key == pygame.K_RIGHT:
                        self.setStride(1)
                    elif event.key == pygame.K_SPACE:
                        self.pause = not self.pause
                        if not self.pause:  # 继续
                            self.startRecv()
                            self.state = "运行中"
                        else:
                            self.state = "暂停"
                    elif event.key == pygame.K_s and pygame.key.get_mods() & pygame.KMOD_CTRL:
                        filename = asksaveasfilename(
                            initialdir="/",
                            title="保存模型",
                            filetypes=(
                                ("torch model files",
                                 "*.tmd"),
                                ("all files",
                                 "*.*")))
                        print(filename)
                        if True == saveModel(filename):
                            self.state = "模型已保存"
                        else:
                            self.state = "模型保存失败"
                    elif event.key == pygame.K_o and pygame.key.get_mods() & pygame.KMOD_CTRL:
                        filename = askopenfilename(
                            initialdir="/",
                            title="加载模型",
                            filetypes=(
                                ("torch model files",
                                 "*.tmd"),
                                ("all files",
                                 "*.*")))
                        print(filename)
                        if True == loadModel(filename):
                            self.state = "模型已加载"
                        else:
                            self.state = "模型加载失败"
                    elif event.key == pygame.K_r and pygame.key.get_mods() & pygame.KMOD_CTRL:
                        pass  # todo restart

            # 显示模型的运行状态、保存加载等信息
            if self.getLabel() == 0 or self.pause:
                self.trainState = "未训练"
            else:
                self.trainState = "训练中"
            pygame.display.set_caption("[FPS]: %.2f  [run state]: %s [train state]: %s" %(self.clock.get_fps(),self.state,self.trainState))
            # pygame.display.set_caption(
            #     "[FPS]: %.2f distance to left boundary: %i to right boundary: %i" %
            #     (self.clock.get_fps(), abs(
            #         self.ball.rect.centerx - self.leftBoundary), abs(
            #         self.rightBoundary - self.ball.rect.centerx)))
            self.allgroup.clear(self.screen, self.background)
            #print(self.label)
            # if self.normalBall == True:
            #     self.normalBall = False
            #     self.ball.image = Ball.grayImage.convert_alpha()

            if self.returnStride:
                ###################################
                # self.ball.image = Ball.image.convert_alpha()
                # self.normalBall = True
                if self.stride:
                    self.ball.update(self.stride, 40)
                ###################################
                '''
                if self.stride:
                    self.A = -2 * self.stride #-A if self.stride > 0 else A
                    self.vt = - self.A #self.stride * SRD / self.T + (1 / 2.0) * self.A * self.T
                    #print('self.vt:', self.vt)
                    # self.ball.update(self.getStride(), SRD)  #
                    # holdOnTime = WT
                else:
                    self.vt = 0
                    self.A = 0
                '''
                self.clearStride()
                self.startRecv()
                print('start recv data')

            # offset = self.getOffsetDis(seconds)
            # print('offset:',offset)
            # self.ball.update(offset)
            # todo 边界碰撞检测

            if self.pause:#todo 最好在非目标间隔内暂停 不然分类器的训练过程会受干扰
                if not self.stopRecvData:
                    self.stopRecv()
            else:
                if isTargetTime:
                    # print('isTargetTime')
                    targetTime += seconds
                    if targetTime > TT:
                        targetTime = 0
                        self.setBlank()
                        isTargetTime = False
                else:
                    #print('not TargetTime')
                    self.setLabel(0)
                    blankTime += seconds
                    if blankTime > BT:
                        blankTime = 0
                        self.setTarget()
                        self.ball.retart()
                        isTargetTime = True

                # holdOnTime -= seconds
                # print(holdOnTime)
                # if holdOnTime <= 0:

            # print(seconds) #todo 保存模型时连训练时长一起保存？

            self.allgroup.draw(self.screen)

            pygame.display.flip()

        pygame.quit()


if __name__ == '__main__':
    from liftData2Classify import trainEEG, TIME_STEP
    from EEGClient import scanTransport
    import queue
    import os
    import torch
    import numpy as np

    game = ballGame()
    classifier = trainEEG()
    transport = scanTransport('159.226.19.2', 4000 , qMaxSize = TIME_STEP)

    testQueue = queue.Queue()

    # def produceEEGsignal(testQueue, getStopRecv, stopRecv):
    #     while True:
    #         if not getStopRecv():
    #             print('begin produce data')
    #             if not testQueue.empty():
    #                 #print('queue not empty')
    #                 testQueue.queue.clear()
    #             for _ in range(TIME_STEP):
    #                 row = np.random.randint(-1000, 1000, size=64)
    #                 testQueue.put(row)
    #             # time.sleep(0.3)
    #             #print('put in queue')
    #             # time.sleep(0.1)
    #             stopRecv()
    #             print('end produce data')
    # def produceStride(getStopRecv,setStride):
    #     looptimes = 20000
    #     while looptimes > 0:
    #         if not getStopRecv():
    #             setStride(random.randint(-1, 1))
    #             looptimes = looptimes - 1
    #             time.sleep(0.1)

    # stopRecvData = [True]
    # stride = [random.uniform(-1, 1)]
    trainOnline = threading.Thread(
        target=classifier.trainOnline, args=(
            transport.eegQueue, game.getLabel, game.setStride,game.stopRecv))
    trainOnline.setDaemon(True)

    # producer = threading.Thread(
    #     target=produceEEGsignal,
    #     args=(
    #         transport.eegQueue,
    #         game.getStopRecv,
    #         game.stopRecv))
    # producer.setDaemon(True)

    scanRecvThread = threading.Thread(
        target=transport.recvData, args=(
            game.getStopRecv, game.stopRecv))

    # tnet = threading.Thread(target=produceStride,args=(game.getRecvData,game.setStride))
    # tnet.setDaemon(True) #设置为后台线程 主线程结束时强制结束
    # tnet.start()
    trainOnline.start()
    #producer.start()
    transport.start()
    scanRecvThread.start()
    # todo 处理网络未连接的异常
    transport.startRecvedEEGData()

    #trainOnline.join()
    game.run(classifier.saveModel,classifier.loadModel)
    transport.disconnect()
    # tnet.join()