# find data from log
import matplotlib.pyplot as plt

x = ['32-bit', '64-bit', '128-bit', '256-bit']
y1 = [9.58, 97.12, 97.72, 97.78]
y2 = [97.5, 97.55, 97.79, 97.82]

plt.plot(x, y1, marker='o')
plt.title('Task 2.2')
plt.xlabel('Communication overhead per view (bits)')
plt.ylabel('Accuracy (%)')
plt.ylim(0, 100)
plt.savefig('bit_vs_value_1.png')
plt.clf()  

plt.plot(x, y2, marker='o')
plt.title('Task 2.3')
plt.xlabel('Communication overhead per view (bits)')
plt.ylabel('Accuracy (%)')
plt.ylim(0, 100)
plt.savefig('bit_vs_value_2.png')
plt.clf() 