import seaborn as sns 
import matplotlib.pyplot as plt
import argparse
import yaml
import utils 
import numpy as np
import sys
from sklearn.preprocessing import StandardScaler
from grakel import Graph
from grakel.kernels import WeisfeilerLehman, VertexHistogram
from grakel.kernels import ShortestPath,WeisfeilerLehmanOptimalAssignment
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from typing import List, Dict,Union
from sklearn.model_selection import GridSearchCV
from collections import defaultdict
import tqdm


def main(config:Dict):
	

	graph_type:str
	vertex_range:Union[str,List[int]]
	graph_data_path:str
	rng_seed:int
	kernels:List[str]
	normalize_kernels:bool
	

	num_splits:int 
	num_iters:int 
	train_size:float

	

	graph_type, vertex_range, graph_data_path, rng_seed, kernels, normalize_kernels =\
		utils.unpack_parameters(config['EXP_PARAMS'])

	num_splits, num_iters, train_size = utils.unpack_parameters(config['CV_PARAMS'])

	
	if isinstance(vertex_range,str):
		num_vertex_min, num_vertex_max = utils.exp_to_bounds[graph_type][vertex_range]
		range_string = vertex_range
	else:
		num_vertex_min, num_vertex_max = vertex_range[0],vertex_range[1]
		range_string = "{min}_{max}".format(min = num_vertex_min, max = num_vertex_max)


	rng = np.random.RandomState(rng_seed)

	existing_features_name= "{dir}/FV{gtype}GraphExistingFeatures.txt".format(dir=graph_data_path, gtype = graph_type)
	existing_features = pd.read_csv(existing_features_name,names = ['ID','S','E','Weight'],sep="\t")
	existing_ids = existing_features['ID'].values


	Graphs = []
	targets = []

	res_path = "../results/{gt}/{r}/".format(gt = graph_type, r= range_string)
	os.makedirs(res_path,exist_ok = True)
	
	for num_vertices in range(num_vertex_min,num_vertex_max+1):
		base_path = "../data/{gt}Graphs/V{n}/".format(n=num_vertices,gt = graph_type)
		Graph_IDs = sorted([f.split(".")[0] for f in os.listdir(base_path)],key = lambda x:int(x.split("_")[1]))

		for gid in Graph_IDs:
			graph_exists = 1 if gid in existing_ids else 0
			
			graph_file = "{base}{id}.txt".format(base=base_path,id=gid)
			AdjM = utils.retrieve_adj_matrix(graph_file)
			
		
			degrees = np.sum(AdjM,axis=1)
			node_labels = {i:degrees[i] for i in range(len(degrees))}
			Graphs.append(Graph(AdjM,node_labels = node_labels))
			targets.append(graph_exists)
	
	
	targets = np.array(targets)
	num_existing = np.where(targets==1)[0].shape[0]
	
	results = defaultdict(list)
	if isinstance(vertex_range,str) or vertex_range == [2,9]:
		positive_samples = np.where(targets==1)[0]
		negative_samples = np.where(targets==0)[0]
		samples = list(np.concatenate( (positive_samples, negative_samples)))
		gk = utils.make_kernel("WeisfeilerLehmanOptimalAssignment", normalize_kernels)
		K = gk.fit_transform([Graphs[i] for i in positive_samples])
		sns.heatmap(K)
		if vertex_range == [2,9]:
			plt.title("2-9")
		else:
			plt.title(vertex_range.title())
		plt.savefig("{d}_existing_heatmap.png".format(d=vertex_range))
		plt.close()


		K = gk.fit_transform([Graphs[i] for i in samples])
		sns.clustermap(K)
		if vertex_range == [2,9]:
			plt.title("2-9")
		else:
			plt.title(vertex_range.title())
		plt.savefig("{d}_heatmap.png".format(d=vertex_range))
		plt.close()
		


	sys.exit(1)
	for s in tqdm.tqdm(range(num_splits)):
	
		negative_samples = np.random.choice(np.where(targets==0)[0],num_existing,replace = False)
		positive_samples = np.where(targets==1)[0]
		
		if isinstance(vertex_range,str) or num_vertex_max<9:
			sample_idxs = [i for i in range(len(targets))]
		else: 
			sample_idxs = np.concatenate((negative_samples,positive_samples))
		
		
		Gs = [Graphs[i] for i in sample_idxs]
		y = [targets[i] for i in sample_idxs]
		for j in tqdm.tqdm(range(num_iters),leave = False):
			
			C_grid = (10. ** np.arange(-4,1,0.1))
			clf = GridSearchCV(SVC(kernel="precomputed"), param_grid = dict(C=C_grid),cv = 5,scoring = 'accuracy')

			
			G_train, G_test, y_train, y_test = train_test_split(Gs, y , train_size = train_size, random_state = rng)
			
			for kernel in kernels:
				gk = utils.make_kernel(kernel, normalize_kernels)
				
				
				K_train = gk.fit_transform(G_train)
				K_test = gk.transform(G_test)
				clf.fit(K_train, y_train)


				test_pred = clf.predict(K_test)
				train_pred = clf.predict(K_train)

				train_acc = accuracy_score(y_train, train_pred)
				test_acc = accuracy_score(y_test, test_pred)

				results['split'].append(s)
				results['iter'].append(j)
				results['kernel'].append(kernel)
				results['normalize'].append(normalize_kernels)
				results['train accuracy'].append(train_acc)
				results['test accuracy'].append(test_acc)


	norm_string = "normalized" if normalize_kernels else "raw"
	results = pd.DataFrame(results)
	results.to_csv(res_path+norm_string+".csv",index=False)
	

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("-config",help="The config file for these experiments")
	args = parser.parse_args()
	
	with open(args.config) as file:
		config = yaml.safe_load(file)

	main(config)

	
	
	
	



	

