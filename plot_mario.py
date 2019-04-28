import matplotlib.pyplot as plt
import csv

x = []
y = []

with open('/home/karthikeya/Documents/courses/AI_Course_project2/deepq/experiments/exp_1/openai-2019-04-26-14-29-54-213770/progress.csv','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    next(csvfile)

    for row in plots:
        x.append(float(row[3]))
        y.append(float(row[2]))
print(x)
plt.plot(x,y, label='Loaded from file!')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Interesting Graph\nCheck it out')
plt.legend()
plt.show()
