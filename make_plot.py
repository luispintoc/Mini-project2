import numpy as np
import matplotlib.pyplot as plt

N = 4
Accuracy = (.768, .712, .816, .856)
ER = (.232, .288, .184, .144)
Precision = (.753,.78899,.83712,.85357)
Recall = (.848,.63704,.81852,.88518)
Specificity = (.673,.8,.813043,.821739)
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
