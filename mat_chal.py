import dill
import matplotlib.pyplot as plt
import matplotlib.animation as animation



A=dill.load(open('solution2', 'rb'))

for c in A:
    plt.imshow(c.reshape((100,100)))
    plt.pause(0.1)