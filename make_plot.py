import numpy as np
import matplotlib.pyplot as plt

N = 4
Accuracy = (.75973,.70346,.82426,.83707)
ER = (.2402,.2965,.1757,.1629)
Precision = (.72526,.75498,.83089,.83518)
Recall = (.838297,.60478,.8154,.84096)
Specificity = (.6807,.80267,.833155,.83316)
ind = np.arange(N)
width = .4

p1 = plt.bar(ind, Accuracy, width)
#p2 = plt.bar(ind, Precision, width)
#p3 = plt.bar(ind, Recall,width)
#p4 = plt.bar(ind, Specificity,width)

plt.figure(1)
plt.ylabel('Accuracy')
plt.xlabel('Features')
plt.title('Accuracy of Bin Naive Bayes')
plt.xticks(ind,('N','P', 'PN', 'PNR'))
plt.yticks(np.arange(0,1,.2))
plt.show()

plt.show()
