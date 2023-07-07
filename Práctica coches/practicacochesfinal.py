
import cv2
import numpy as np 
import matplotlib.pyplot as plt
import Cars


#Background.
result = None
cap = cv2.VideoCapture('trafico.mp4')

FOI = cap.get(cv2.CAP_PROP_FRAME_COUNT) * np.random.uniform(size=30)
frames = []
for frameOI in FOI:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frameOI)

    ret, frame = cap.read()
    frames.append(frame)

result = np.median(frames, axis=0).astype(dtype=np.uint8)
background = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

#Car counter.
cap = cv2.VideoCapture('trafico.mp4')
car_counter = 0
t = 0
intervalo = 9
cars = []
succ, frame = cap.read()
i = 0
while succ:

    succ, frame = cap.read()
    if not succ:
        break
    t += 1
    intervalo -= 1

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    backgroundgray = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)

    sub = cv2.subtract(gray, backgroundgray)

    th = np.max(sub) * 0.5
    ret, thresh1 = cv2.threshold(sub, th, 255, cv2.THRESH_BINARY)

    kernel = np.ones((8,8), np.uint8) 
    img_dilation = cv2.dilate(thresh1, kernel, iterations=1) 

    contours, _ = cv2.findContours(img_dilation, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    filterContorurs = []
    centers = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        if x < 800:
            filterContorurs.append(cnt)
            c = Cars.center(x,y,w,h)
            centers.append(c)
    
    newCont = []
    
    cntAnt = filterContorurs[0]
    centerAnt= centers[0]
    i = 0

    while i < len(centers):
        dist = Cars.distancia(centerAnt, centers[i])
        if dist < 60:
            x1,y1,w1,h1 = cv2.boundingRect(cntAnt)
            x,y,w,h = cv2.boundingRect(filterContorurs[i])
            if x > x1 and (x+w) < (x1 + w1):
                if y > y1 and (y1+h1) > (y + h):
                    newCont.append([x1,y1,w1,h1])
                    cntAnt = filterContorurs[i]
                    centerAnt = centers[i]
                    i += 1
            else:
                wnew = max(x+w, x1+w1) - min(x,x1)
                hnew = max(y+h, y1+h1) - min(y, y1)
                newCont.append([min(x1,x), min(y1,y), wnew , hnew])
                cntAnt = filterContorurs[i]
                centerAnt = centers[i]
                i += 1
        else:
            cntAnt = filterContorurs[i]
            centerAnt = centers[i]
        i += 1

    for cnt in newCont:
        center = Cars.center(cnt[0],cnt[1],cnt[2],cnt[3])
        
        #Añadir un nuevo coche.
        if t == 9 or t==8 :
            if 450 < center[0] < 600 and 470 < center[1] < 510:
                car_counter += 1
                template = frame[cnt[1]: (cnt[1]+cnt[3]), cnt[0]: (cnt[0]+cnt[2])]
                c = Cars.Cars(car_counter, center, template)
                cars.append(c)
                t = -3
                intervalo = 9
            
        #Actualizar centro y template.
        else:
            for c in cars:
                dist = Cars.distancia(center, c.getCenter())
                if dist < 60 and dist != 0:
                    newTemplate = frame[cnt[1]: (cnt[1]+cnt[3]), cnt[0]: (cnt[0]+cnt[2])]
                    c.setCenter(center)
                    c.setTemplate(newTemplate)
                    if 450 < int(c.getCenter()[0]-100) < 600 and 470 < int(c.getCenter()[1]-100) < 670:
                        cv2.putText(frame, "Id: " + str(c.getId()), (int(c.getCenter()[0]-50),int(c.getCenter()[1]-50)), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=1, color=(0, 0, 255),thickness=2)

        res = cv2.rectangle(frame, (cnt[0], cnt[1]), (cnt[0]+cnt[2], cnt[1]+cnt[3]), (255, 0, 0), 2)
        res = cv2.circle(frame, (round(center[0]), round(center[1])), 2, (0,0,255), 3)
        cv2.line(frame, (450, 495), (600, 495), (0, 255, 255), 2)

    for car in cars:
        if car.getCenter()[1] < 1000:
            cv2.rectangle(frame,(int(car.getCenter()[0]-100), int(car.getCenter()[1]-100)), (int(car.getCenter()[0]+100), int(car.getCenter()[1]+100)), (0,0,255), 3)
        frame2 = frame[int(car.getCenter()[1]-100):int(car.getCenter()[1]+100), int(car.getCenter()[0]-100): int(car.getCenter()[0]+100)]
        cv2.imshow('', frame2)
        res = cv2.matchTemplate(frame2,car.getTemplate(),0)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        xmin, ymin = min_loc
        xmax, ymax = min_loc[0] + template.shape[1], min_loc[1] + template.shape[0]
        if car.getCenter()[1] < 1000:
            cv2.rectangle(frame2,(xmin,ymin),(xmax,ymax),(59,219,137),2)
        
    if intervalo == 0:
        t = 0
        intervalo = 9

        

    area_pts = np.array([[400,900], [800, 900],[800,700],[400,700]])
    cv2.putText(frame, "Car counter: " + str(car_counter), (400,100), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=2, color=(0, 0, 255),thickness=3)
    cv2.imshow('frame', frame)

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

    elif k == ord('q'):
        cv2.waitKey()
    
print(car_counter)
cap.release()
cv2.destroyAllWindows()
