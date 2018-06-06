import pygame
import EEGClient
import os,threading,random,time
from tkinter.filedialog import asksaveasfilename, askopenfilename
#os.environ['SDL_VIDEO_WINDOW_POS'] = str(position[0]) + "," + str(position[1])
os.environ['SDL_VIDEO_CENTERED'] = '1'  # 窗口中央显示

#####################################
TT = 3  #target time limit (s)
BT = 1  #blank time limit  (s)
WT = 0.1  #watch ball time (s)
########################################

class Ball(pygame.sprite.Sprite):
    image = pygame.image.load("ball.png")
    image.set_colorkey((0, 0, 0))
    #image = image.convert_alpha()
    #centerPos = (SCREENWIDTH/2,SCREENHEIGHT/2)

    def __init__(self, startpos):
        # 获取系统参数如屏幕
        pygame.sprite.Sprite.__init__(self, self.groups)
        self.pos = startpos
        self.image = Ball.image.convert_alpha()
        self.rect = self.image.get_rect()
        self.rect.center = self.pos

    def update(self, direction, stride=1):
        '''
        direction:-1到1之间的参数
        stride:小球一次运动的步长
        '''
        self.rect.centerx += direction * stride

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
        self.pause = True

        self.ballgroup = pygame.sprite.Group()
        self.allgroup = pygame.sprite.Group()
        Ball.groups = self.ballgroup, self.allgroup  # 这么设计真的好吗
        TargetArea.groups = self.allgroup

        self.ball = Ball((self.CENTER_X, self.CENTER_Y))

        self.leftBoundary = self.CENTER_X - self.INTERVAL - TargetArea.areaWidth / 2
        self.rightBoundary = self.CENTER_X + self.INTERVAL - TargetArea.areaWidth / 2
        self.topBoundary = self.CENTER_Y - TargetArea.areaHeight / 2
        self.leftArea = TargetArea(self.leftBoundary, self.topBoundary)
        self.rightArea = TargetArea(self.rightBoundary, self.topBoundary)

        self.strideLock = threading.Lock()
        self.stopRecvLock = threading.Lock()
        self.keepOn = False
        #self.trans = EEGClient.scanTransport(remoteHost, remotePort)

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
        self.leftArea.update(1)

    def setBlank(self):
        self.leftArea.update(False)
        self.rightArea.update(False)
        #print('setBlank')

    def setTarget(self):
        r = random.randint(0,1)
        self.rightArea.update(r)
        self.leftArea.update(1-r)
        #print('setTarget')

    def run(self, stride, stopRecvData):
        self.initGameWindow()

        continued = 0
        isTargetTime = True
        targetTime = 0
        blankTime = 0
        holdOnTime = 0


        while self.mainloop:
            # milliseconds passed since last frame
            milliseconds = self.clock.tick(self.FPS)
            seconds = milliseconds / 1000.0  # seconds passed since last frame
            continued += seconds
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.mainloop = False  # pygame window closed by user
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.mainloop = False  # user pressed ESC
                    elif event.key == pygame.K_LEFT:
                        self.ball.update(-1, 30)
                    elif event.key == pygame.K_RIGHT:
                        self.ball.update(1, 30)
                    elif event.key == pygame.K_SPACE:
                        self.pause = not self.pause
                    elif event.key == pygame.K_s and pygame.key.get_mods() & pygame.KMOD_CTRL:
                        filename = asksaveasfilename(
                            initialdir="/",
                            title="保存模型",
                            filetypes=(
                                ("jpeg files",
                                 "*.jpg"),
                                ("all files",
                                 "*.*")))
                        print(filename)
                        # todo save netmodal
                    elif event.key == pygame.K_o and pygame.key.get_mods() & pygame.KMOD_CTRL:
                        filename = askopenfilename(
                            initialdir="/",
                            title="加载模型",
                            filetypes=(
                                ("jpeg files",
                                 "*.jpg"),
                                ("all files",
                                 "*.*")))
                        print(filename)
                        # todo load netmodal
                    elif event.key == pygame.K_r and pygame.key.get_mods() & pygame.KMOD_CTRL:
                        pass  # todo restart

            # todo 可显示模型的保存加载信息状态
            pygame.display.set_caption(
                "[FPS]: %.2f distance to left boundary: %i to right boundary: %i" %
                (self.clock.get_fps(), abs(
                    self.ball.rect.centerx - self.leftBoundary), abs(
                    self.rightBoundary - self.ball.rect.centerx)))
            self.allgroup.clear(self.screen, self.background)

            if stride[0]:  # todo 放到if pause 外面

                self.ball.update(stride[0], 10)  # todo  将random改为stride[0],即网络的输出[-1,1],边界碰撞检测
                self.strideLock.acquire()
                stride[0] = 0
                self.strideLock.release()
                holdOnTime = WT

            if self.pause:
                self.stopRecvLock.acquire()
                #print('get stopRecvData')
                stopRecvData[0] = True #todo 每次向队列里放元素时先置空
                self.stopRecvLock.release()
            else:
                if isTargetTime:
                    targetTime += seconds
                    if targetTime > TT:
                        targetTime = 0
                        self.setBlank()
                        isTargetTime = False
                else:
                    blankTime += seconds
                    if blankTime > BT:
                        blankTime = 0
                        self.setTarget()
                        self.ball.retart()
                        isTargetTime = True

                holdOnTime -= seconds
                #print(holdOnTime)
                if holdOnTime <= 0:
                    self.stopRecvLock.acquire()
                    stopRecvData[0] = False
                    self.stopRecvLock.release()


            #print(continued) #todo 保存模型时连训练时长一起保存？

            self.allgroup.draw(self.screen)

            pygame.display.flip()

        pygame.quit()


if __name__ == '__main__':

    game = ballGame()

    def produceStride(stride,stopRecvData):
        looptimes = 20000
        while looptimes > 0:
            if not stopRecvData[0]:
                game.strideLock.acquire()
                stride[0] = random.randint(-1, 1)
                game.strideLock.release()
                looptimes = looptimes - 1
                time.sleep(0.1)



    stopRecvData = [True]
    stride = [random.uniform(-1, 1)]

    tnet = threading.Thread(target=produceStride,args=(stride,stopRecvData))

    tnet.start()
    game.run(stride,stopRecvData)
    #tnet.join()