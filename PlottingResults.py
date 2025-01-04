np.savez( "PlotDataR2" + ".npz", Loss=LossList, Epochs=EpochList)

fig, ax = plt.subplots()
ax.plot(EpochList, LossList)
ax.set_xlabel("Epochs")
ax.set_ylabel("Loss")
ax.set_title("Loss function value over epochs (1200 samples)")
plt.show()