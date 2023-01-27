import dill
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pygame as pg


A=dill.load(open('solution', 'rb'))


pg.init()
screen = pg.display.set_mode((500,500 ))
clock = pg.time.Clock()
m=0
running=True
while m<len(A) and running:
    r=A[m].reshape(100,100)
    for i in range(len(A)):
        for j in range(len(A[i])):
            rect = pg.Rect(i*5, j*5, 5, 5)
            pg.draw.rect(screen, color, rect)
    m+=1
    pg.display.update()
    for event in pg.event.get():
        if event.type == pg.QUIT:
            running = False
pg.quit()