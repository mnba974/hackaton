import dill
import matplotlib.pyplot as plt
import matplotlib.animation as animation



A=dill.load(open('solution', 'rb'))
print(A)
def animate(i):
    return A[i].reshape((100,100))
fig=plt.figure()
ani = animation.FuncAnimation(fig, animate, frames=100,interval=1, blit=True, repeat=False)
plt.show()