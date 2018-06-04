import pygame,random,os
from tkinter.filedialog import asksaveasfilename,askopenfilename
#os.environ['SDL_VIDEO_WINDOW_POS'] = str(position[0]) + "," + str(position[1])
os.environ['SDL_VIDEO_CENTERED'] = '1'

#pygame.init()
'''
disInfo = pygame.display.Info()
print(disInfo)
SCREENWIDTH = disInfo.current_w
SCREENHEIGHT = disInfo.current_h
BACKGROUNDCOLOR = (255,255,255)
INTERVAL = 500
CENTER_X = SCREENWIDTH/2
CENTER_Y = SCREENHEIGHT/2
'''

#screen = pygame.display.set_mode((SCREENWIDTH,SCREENHEIGHT),pygame.RESIZABLE) #pygame.FULLSCREEN
'''
def write(msg="pygame is cool"):
    myfont = pygame.font.SysFont("微软雅黑", 32)
    mytext = myfont.render(msg, True, (0,0,0))
    mytext = mytext.convert_alpha()
    return mytext

background = pygame.Surface(screen.get_size())
background.fill(BACKGROUNDCOLOR)     # fill white
background.blit(write('+'),(CENTER_X,CENTER_Y))
# background.blit(write("Press ESC to quit"),(300,10))
background = background.convert()  # jpg can not have transparency
screen.blit(background, (0,0))     # blit background on screen (overwriting all)
clock = pygame.time.Clock()        # create pygame clock object
mainloop = True
FPS = 60                           # desired max. framerate in frames per second.
'''


class Ball(pygame.sprite.Sprite):
    image = pygame.image.load("ball.png")
    image.set_colorkey((0,0,0))
    #image = image.convert_alpha()
    #centerPos = (SCREENWIDTH/2,SCREENHEIGHT/2)
    def  __init__(self,startpos):
        #获取系统参数如屏幕
        pygame.sprite.Sprite.__init__(self,self.groups)
        self.pos = startpos
        self.image = Ball.image.convert_alpha()
        self.rect = self.image.get_rect()
        self.rect.center = self.pos

    def update(self,direction,stride=1):
        '''
        direction:-1到1之间的参数
        stride:小球一次运动的步长
        '''
        self.rect.centerx += direction * stride

    def retart(self):
        self.rect.center = self.pos #Ball.centerPos

class TargetArea(pygame.sprite.Sprite):
    areaWidth = 200
    areaHeight = 600
    defaultColor = (200,200,200)
    targetColor = (0,150,0)
    def __init__(self, left,top, width = areaWidth,height = areaHeight):
        pygame.sprite.Sprite.__init__(self, self.groups)
        self.left = left
        self.top = top
        self.width = width
        self.height = height
        self.image = pygame.Surface((width,height))
        self.image.fill(TargetArea.defaultColor)
        self.rect = self.image.get_rect()
        self.updateColor = TargetArea.defaultColor
        self.rect.center = (width/2+left,height/2+top)

    def update(self, target):
        if target > 0:#边框（区域）变为亮色
            self.updateColor = TargetArea.targetColor
        else:#灰色
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
        self.screen = pygame.display.set_mode((self.SCREENWIDTH,self.SCREENHEIGHT),pygame.RESIZABLE) #pygame.FULLSCREEN
        self.background = pygame.Surface(self.screen.get_size())

        self.clock = pygame.time.Clock()  # create pygame clock object
        self.mainloop = True
        self.FPS = 60  # desired max. framerate in frames per second.
        self.pause = True

        self.ballgroup = pygame.sprite.Group()
        self.allgroup = pygame.sprite.Group()
        Ball.groups = self.ballgroup, self.allgroup  # 这么设计真的好吗
        TargetArea.groups = self.allgroup

        self.ball = Ball((self.CENTER_X,self.CENTER_Y))

        self.leftBoundary = self.CENTER_X - self.INTERVAL - TargetArea.areaWidth / 2
        self.rightBoundary = self.CENTER_X + self.INTERVAL - TargetArea.areaWidth / 2
        self.topBoundary = self.CENTER_Y - TargetArea.areaHeight / 2
        self.leftArea = TargetArea(self.leftBoundary, self.topBoundary)
        self.rightArea = TargetArea(self.rightBoundary, self.topBoundary)



    def write(self,msg="ball game"):
        myfont = pygame.font.SysFont("微软雅黑", 32)
        mytext = myfont.render(msg, True, (0, 0, 0))
        mytext = mytext.convert_alpha()
        return mytext

    def initGameWindow(self):
        self.background.fill(self.BACKGROUNDCOLOR)  # fill white
        self.background.blit(self.write('+'), (self.CENTER_X, self.CENTER_Y))
        # background.blit(write("Press ESC to quit"),(300,10))
        self.background = self.background.convert()  # jpg can not have transparency
        self.screen.blit(self.background, (0, 0))  # blit background on screen (overwriting all)
        self.leftArea.update(1)

    def run(self):
        self.initGameWindow()

        continued = 0
        trialInterval = range(3, 7)
        randomInterVal = 0
        lastTarget = -1

        while self.mainloop:
            milliseconds = self.clock.tick(self.FPS)  # milliseconds passed since last frame
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
                        filename = asksaveasfilename(initialdir="/", title="保存模型",
                                                     filetypes=(("jpeg files", "*.jpg"), ("all files", "*.*")))
                        print(filename)
                        # todo save netmodal
                    elif event.key == pygame.K_o and pygame.key.get_mods() & pygame.KMOD_CTRL:
                        filename = asksaveasfilename(initialdir="/", title="加载模型",
                                                     filetypes=(("jpeg files", "*.jpg"), ("all files", "*.*")))
                        print(filename)
                        # todo load netmodal
                    elif event.key == pygame.K_r and pygame.key.get_mods() & pygame.KMOD_CTRL:
                        pass  # todo restart

            # todo 可显示模型的保存加载信息状态
            pygame.display.set_caption(
                "[FPS]: %.2f distance to left boundary: %i to right boundary: %i"
                % (self.clock.get_fps(),abs(self.ball.rect.centerx - self.leftBoundary),abs(self.rightBoundary - self.ball.rect.centerx)))
            self.allgroup.clear(self.screen, self.background)

            if not self.pause:
                self.ball.update(random.randint(-1, 1))

                if continued > trialInterval[randomInterVal]:  # todo 边界碰撞检测

                    target = -lastTarget
                    lastTarget = target
                    self.ball.retart()
                    self.leftArea.update(-target)
                    self.rightArea.update(target)
                    continued = 0
                    randomInterVal = random.randint(0, len(trialInterval) - 1)

            self.allgroup.draw(self.screen)

            pygame.display.flip()

        pygame.quit()


if __name__ == '__main__':
    game = ballGame()
    game.run()

'''
ballgroup = pygame.sprite.Group()
allgroup = pygame.sprite.Group()
Ball.groups = ballgroup,allgroup  #这么设计真的好吗
TargetArea.groups = allgroup

ball = Ball(Ball.centerPos)
leftBoundary = CENTER_X -INTERVAL - TargetArea.areaWidth/2
rightBoundary = CENTER_X + INTERVAL - TargetArea.areaWidth/2
topBoundary = CENTER_Y-TargetArea.areaHeight/2

leftArea = TargetArea(leftBoundary,topBoundary)
rightArea = TargetArea(rightBoundary,topBoundary)

leftArea.update(1)

continued = 0
trialInterval = range(3,7)
randomInterVal = 0
lastTarget = -1
pause = True

while mainloop:
    milliseconds = clock.tick(FPS)  # milliseconds passed since last frame
    seconds = milliseconds / 1000.0  # seconds passed since last frame
    continued += seconds
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            mainloop = False  # pygame window closed by user
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                mainloop = False  # user pressed ESC
            elif event.key == pygame.K_LEFT:
                ball.update(-1,30)
            elif event.key == pygame.K_RIGHT:
                ball.update(1,30)
            elif event.key == pygame.K_SPACE:
                pause = not pause
            elif event.key == pygame.K_s and pygame.key.get_mods() & pygame.KMOD_CTRL:
                filename = asksaveasfilename(initialdir = "/",title = "保存模型",filetypes = (("jpeg files","*.jpg"),("all files","*.*")))
                print(filename)
                #todo save netmodal
            elif event.key == pygame.K_o and pygame.key.get_mods() & pygame.KMOD_CTRL:
                filename = asksaveasfilename(initialdir = "/",title = "加载模型",filetypes = (("jpeg files","*.jpg"),("all files","*.*")))
                print(filename)
                #todo load netmodal
            elif event.key == pygame.K_r and pygame.key.get_mods() & pygame.KMOD_CTRL:
                pass #todo restart

    #todo 可显示模型的保存加载信息状态
    pygame.display.set_caption("[FPS]: %.2f distance to left boundary: %i to right boundary: %i" % (clock.get_fps(),
                                                                                                    abs(ball.rect.centerx-leftBoundary),
                                                                                                    abs(rightBoundary-ball.rect.centerx)))
    allgroup.clear(screen, background)

    if not pause:
        ball.update(random.randint(-1,1))

        if continued > trialInterval[randomInterVal]: #todo 边界碰撞检测

            target = -lastTarget
            lastTarget = target
            ball.retart()
            leftArea.update(-target)
            rightArea.update(target)
            continued = 0
            randomInterVal = random.randint(0, len(trialInterval) - 1)

    allgroup.draw(screen)

    pygame.display.flip()
'''