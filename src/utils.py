from grakel.kernels import WeisfeilerLehman, VertexHistogram
from grakel.kernels import ShortestPath,WeisfeilerLehmanOptimalAssignment
import numpy as np 
from typing import Dict


exp_to_bounds = {'Tree':{'lower':(2,5), 'upper':(7,13)}, 'Dual':{'lower':(2,5), 'upper':(6,9),'pooled':(2,9)}}
VIRAL_TRNA = ['3_6','4_17', '5_4','5_5','6_2','7_1']
VIRAL_FS = ['2_3','3_8','4_20']
snoRNA = ['4_4']

def retrieve_adj_matrix(
	fname:str
	)-> np.ndarray:
	with open(fname,"r") as istream:
		lines = istream.readlines()
		lines = [x.rstrip() for x in lines[1:]]
		lines = [[int(y) for y in p.split()] for p in lines]
	AdjM = np.array(lines)

	return AdjM

def unpack_parameters(D:Dict):
	if len(D.values())>1:
		return tuple(D.values())
	else:
		return tuple(D.values())[0]

def make_kernel(
	name:str,
	normalize:bool
	):
	

	if name == "ShortestPath":
		gk = ShortestPath(normalize = normalize)
	elif name == "VertexHistogram":
		gk = VertexHistogram(normalize = normalize)
	elif name == "WeisfeilerLehman":
		gk = WeisfeilerLehman(normalize = normalize)
	elif name == "WeisfeilerLehmanOptimalAssignment":
		gk = WeisfeilerLehmanOptimalAssignment(normalize = normalize)

	return gk