import math
class Cars:

    def __init__(self, id, center, template):
        self.id = id
        self.center = center
        self.template = template

    def getId(self):
        return self.id

    def getCenter(self):
        return self.center

    def getTemplate(self):
        return self.template
    
    def setCenter(self, newCenter):
        self.center = newCenter
    
    def setTemplate(self, newTemplate):
        self.template = newTemplate
    

def center(x, y, w, h):
    x1 = (2*x+h)/2
    y1 = (2*y+w)/2
    return (x1,y1)

def distancia(centerAnt, center):
    dist = ((centerAnt[0]-center[0])**2 + (centerAnt[1] - center[1])**2)**0.5
    return dist