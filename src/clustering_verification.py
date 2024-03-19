import sys
import pandas as pd
import numpy as np 
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

from functools import partial


def assign_group(cluster, label, RNA_like):
	if label ==1:
		group = "Existing"
	elif cluster == RNA_like:
		group = "RNA-Like"
	else:
		group = "Not-RNA-Like"
	return group 


def main():
	seed = 1234
	rng = np.random.RandomState(seed)

	existing_graphs = pd.read_csv("../data/feature_train_qiyao.csv")
	test_graphs = pd.read_csv("../data/feature_test_qiyao.csv")

	existing_graphs['label'] = 1
	test_graphs['label'] = 0 
	all_graphs = pd.concat([existing_graphs,test_graphs])

	all_features = all_graphs[['S','E']].values

	clustering = KMeans(n_clusters = 2, n_init = 'auto',random_state = rng).fit(all_features)

	all_graphs['KMeans_Label'] = clustering.labels_

	# clustering = AffinityPropagation(random_state=rng).fit(all_features)

	# all_graphs['AffProp'] = clustering.labels_


	
	label_stats = pd.DataFrame(all_graphs['KMeans_Label'].value_counts()).reset_index()
	label_stats.to_csv("../results/Kmeans_label_counts.csv",index = False)
	
	counts = []
	
	for cluster_id in [0,1]:
		num_existing = np.sum(all_graphs[all_graphs['KMeans_Label']==cluster_id]['label'].values)
		counts.append(num_existing)
	
	RNA_like= np.argmax(counts)

	assign_grp = partial(assign_group,RNA_like = RNA_like)
	
	acc = np.round(counts[RNA_like]/existing_graphs.shape[0],4)
	
	with open("../results/kmeans_accuracy.txt","w") as ostream:
		ostream.writelines(str(acc))

	all_graphs['Class'] = all_graphs.apply(lambda x: assign_grp(x.KMeans_Label,x.label),axis=1)
	
	ax = sns.scatterplot(all_graphs, x='S',y='E',hue = 'Class', style = 'Class', size = 'Class', palette = ['red','blue','black'])
	sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
	plt.tight_layout()
	plt.savefig("../figs/KNN_Test.png")
	plt.close()


	ex_graph = all_graphs[all_graphs['label']==1]
	ax = sns.scatterplot(ex_graph, x='S',y='E',hue = 'Class',size = 'Class', palette = ['red'])
	sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
	plt.tight_layout()
	plt.savefig("../figs/KNN_Existing.png")
	plt.close()


	



	







if __name__ == '__main__':
	main()