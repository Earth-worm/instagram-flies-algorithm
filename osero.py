import numpy as np

class Osero:
    Width = 8
    Height = 8

    #white:0, black:1, empty:-1
    Field = None

    def __init__(self):
        self.Field = [[-1 for i in range(self.Width)] for i in range(self.Height)]
        self.Field[3][3] = 0
        self.Field[3][4] = 1
        self.Field[4][3] = 1
        self.Field[4][4] = 0

    def Disp(self):
        s = "  "
        for i in range(self.Width):
            s += "|"+str(i)+" "
        s += "\n  "
        for i in range(self.Width):
            s += "|--"
        s += "\n"
        for i in range(self.Height):
            s += str(i) + " "
            for j in range(self.Width):
                if self.Field[i][j] == 0:
                    s += "|● "
                if self.Field[i][j] == 1:
                    s += "|○ "
                if self.Field[i][j] == -1:
                    s += "|  "
            s += "|\n"
        s+= "  "
        for i in range(self.Width):
            s += "|--"
        s += "\n"
        print(s)

    def Put(self,y,x,color):
        dxArr = [0,1,1,1,0,-1,-1,-1]
        dyArr = [1,1,0,-1,-1,-1,0,1]
        for dx,dy in zip(dxArr,dyArr):
            if(l := self.CheckDepth(color,y,x,dy,dx),l!=-1):
                xx = x
                yy = y
                for i in range(l+1):
                    self.Field[yy][xx] = color
                    xx += dx
                    yy += dy

    def CheckDepth(self,color,y,x,dy,dx):
        depth = 0
        while(True):
            y += dy
            x += dx
            if(self.IsInside(y,x)==False or self.Field[y][x] == -1):
                return -1
            if(self.Field[y][x] == color):
                return depth
            depth += 1

    def IsInside(self,y,x):
        return x >= 0 and x < self.Width and y >= 0 and y < self.Height

    def CanPut(self,y,x,color):
        if not self.IsInside(y, x) or self.Field[y][x] != -1:
            return False
        flag = False
        dxArr = [0,1,1,1,0,-1,-1,-1]
        dyArr = [1,1,0,-1,-1,-1,0,1]
        for dx,dy in zip(dxArr,dyArr):
            if(self.CheckDepth(color,y,x,dy,dx) > 0):
                return True

    def GetPossiblePutPositionAndValue(self,color,stoneMap):
        positions = []
        values = []
        for y in range(self.Height):
            for x in range(self.Width):
                if self.CanPut(y,x,color):
                    positions.append([y,x])
                    values.append(self.CalcStoneValue(y,x,color,stoneMap))
        return positions,values

    # stoneMap: array(width * height)
    def CalcStoneValue(self,y,x,color,stoneMap):
        dxArr = [0,1,1,1,0,-1,-1,-1]
        dyArr = [1,1,0,-1,-1,-1,0,1]
        value = stoneMap[y][x]
        for dx,dy in zip(dxArr,dyArr):
            if(l := self.CheckDepth(color,y,x,dy,dx),l!=-1):
                xx = x+dx
                yy = y+dy
                for _ in range(l):
                    if self.Field[yy][xx] != color:
                        value += stoneMap[yy][xx]
                    xx += dx
                    yy += dy
        return value
    
    def Result(self):
        n_white = 0
        n_black = 0
        n_sum = 0
        for y in range(self.Height):
            for x in range(self.Width):
                if self.Field[y][x] == -1:
                    continue
                n_sum += 1
                if self.Field[y][x] == 0:
                    n_white += 1
                if self.Field[y][x] == 1:
                    n_black += 1
        return n_white,n_black,n_sum

def PlayOsero(stoneMap1,stoneMap2):
    o = Osero()
    flag = False
    color = 0
    while(True):
        stoneMap = None
        if color == 0:
            stoneMap = stoneMap1
        else:
            stoneMap = stoneMap2
        arr,values = o.GetPossiblePutPositionAndValue(color,stoneMap2)

        ## 二連続でどっちも置く場所がなければ終わり
        if len(arr) == 0:
            if flag:
                break
            color = (color + 1)%2
            flag = True
            continue
        else:
            flag = False
        best_hand_arg = np.argmax(values)
        best_hand = arr[best_hand_arg]
        o.Put(best_hand[0],best_hand[1],color)
        color = (color + 1)%2
    return o.Result()

if __name__ == '__main__':
    o = Osero()
    color = 0
    """
    stoneMap = [
            [30,-12,0,-1,-1,0,-12,30],
            [-12,-15,-3,-3,-3,-15,-15,-12],
            [0,-3,0,-1,-1,0,-3,0],
            [-1,-3,-1,-1,-1,-1,-3,-1],
            [-1,-3,-1,-1,-1,-1,-3,-1],
            [0,-3,0,-1,-1,0,-3,0],
            [-12,-15,-3,-3,-3,-15,-15,-12],
            [30,-12,0,-1,-1,0,-12,30],
            ]
    """
    stoneMap = [[  4.64442892,-5.94676174  ,-8.22406289,-3.54108864,14.449 -12.25701246, -4.85005069, -3.0631509 ],
     [  2.44601732 , 7.28178502,  3.87052249, -6.61112403 , 2.88436812, 2.28200454 , 2.88739893 , 6.69401217],
     [ -9.63874628 , 1.33511466 , 3.94920367 , 4.25621057 , 8.48590166, -18.34996554,  4.75598223  -5.86846399],
     [-21.5525791 ,-10.40855294,  9.65304426 ,-6.7865024  , 3.43632992,1.78858655,  8.07561675, 3.20647401],
     [  3.35587021 ,-5.91740211, -3.10004183, 16.31703601, -2.99102139,4.97680409,  4.27546672, -2.95882175],
     [  1.82883348 ,-3.98875031 , 3.50467481  ,5.20606844, -0.58939011, -6.0727421,  2.78332894, -4.35495494],
     [ -8.55768735,-5.81515668, 3.06846976 , -4.45747499 , -2.30405207 ,-4.10293077 ,-4.88844761 , 5.65605183],
     [  0.51941962, -1.71454956 ,-6.73718104, -6.2447815  , 3.09940919 ,5.71136616 ,-4.03695552 ,-1.24283584]]
    flag = False
    while(True):
        if color == 0:
            print("●の番です")
        else:
            print("○の番です")
        o.Disp()
        arr,values = o.GetPossiblePutPositionAndValue(color,stoneMap)
        ## 二連続でどっちも置く場所がなければ終わり
        if len(arr) == 0:
            if flag:
                break
            color = (color + 1)%2
            flag = True
            continue
        else:
            flag = False
        print("おける場所")
        for (yy,xx) in arr:
            print("yy:",yy,"xx:",xx,"value:",o.CalcStoneValue(yy,xx,color,stoneMap))
        y = int(input("y:"))
        x = int(input("x:"))
        if not o.CanPut(y, x, color):
            print("置やんぞ!!")
        else:
            o.Put(y,x,color)
            color = (color + 1)%2
    n_white,n_black,n_sum = o.Result()
    print("white:",n_white,"black:",n_black,"total:",n_sum)
