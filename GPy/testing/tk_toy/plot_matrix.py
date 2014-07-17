import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import sys

font = {'family' : 'sans-serif',
        'weight' : 'normal',
        'size'   : 16}

matplotlib.rc('font', **font)

fig, ax = plt.subplots()
image = np.loadtxt(sys.argv[1])

#image += 5
#image *= 100
#image /= 200
#plt.imshow(image, cmap=plt.cm.gray, interpolation='nearest')
plt.imshow(image, cmap=plt.cm.jet, interpolation='nearest')
#plt.imshow(image, cmap=plt.cm.gray, interpolation='hamming')
#plt.colorbar()

#imgplot = plt.imshow(image, interpolation='hanning')
#imgplot = plt.imshow(image, interpolation='none')
#imgplot.set_cmap('jet')
plt.colorbar()
#ax = imgplot.add_subplot(111)


# Move left and bottom spines outward by 10 points
#ax.spines['left'].set_position(('outward', 10))
#ax.spines['bottom'].set_position(('outward', 10))
# Hide the right and top spines
#ax.spines['right'].set_visible(False)
#ax.spines['top'].set_visible(False)
# Only show ticks on the left and bottom spines
#EMOS = ['sadness','fear','anger','disgust','surprise','joy']
EMOS = ['fear','sadness','anger','disgust','surprise','joy']
#ax.yaxis.set_ticks_position('left')
#ax.xaxis.set_ticks_position('bottom')

#labels = [item.get_text() for item in ax.get_xticklabels()]
#labels
#ax.xaxis.set_ticks(np.arange(-0.5,5.6,1))
#ax.yaxis.set_ticks(np.arange(-0.5,5.6,1))
#ax.xaxis.set_ticks(np.arange(0,6,1), minor=True)
#ax.yaxis.set_ticks(np.arange(0,6,1), minor=True)
#ax.set_xticklabels(EMOS, minor=True, rotation=20)
#ax.set_yticklabels(EMOS, minor=True)
#ax.set_xticklabels([])
#ax.set_yticklabels([])
#ax.grid(which='major', axis='both', linestyle='-')
plt.show()
