%{
    
Code to read in the dual graphs and output them 
into individual adjacency matrix files

}%
data = load('../data/FVTreeVertexGraphCounts.txt');

fileID = fopen('../data/TreeGraphID.txt','r');
lines = textscan(fileID,'%s');
mkdir("../data/TreeGraphs/")
GraphIDs = lines{1};

id_ptr = 1;

fclose(fileID);

for i = 1 : size(data,1) % for every vertex number
   
    num_graphs = data(i,1);    
    
    N = data(i,1); % number of vertices
    subdir = strcat("../data/TreeGraphs/V", num2str(N));
    mkdir(subdir)
    num_graphs = data(i,2);
    curr_ids = GraphIDs(id_ptr:id_ptr+num_graphs-1);
    printf("working on tree graphs with %d vertices\n", N)
    printf("There are %d such graphs\n",num_graphs)
  
    graph_file = sprintf('../data/TreeAdj/V%dAdjDG', N);
    matrices = load(graph_file);
    
    for j = 1 : size(matrices, 1)/N
        graph_id = curr_ids(j){1};
        AdjM = matrices((j-1)*N+1:j*N, :);
        
        opath = strcat(subdir,"/",graph_id,".txt");
        disp(opath);
        disp(graph_id);
        file_id = fopen(opath, 'w');
        fdisp(file_id,graph_id);
        fdisp(file_id,AdjM);
        fclose(file_id);
    end
    id_ptr = id_ptr+num_graphs;
end
clc;
clear all;

data = load('../data/FVDualVertexGraphCounts.txt');


fileID = fopen('../data/DualGraphID.txt','r');
lines = textscan(fileID,'%s');
mkdir("../data/DualGraphs/")
GraphIDs = lines{1};

id_ptr = 1;

fclose(fileID);
for i = 1 : size(data,1) % for every vertex number
   
    num_graphs = data(i,1);    
    
    N = data(i,1); % number of vertices
    subdir = strcat("../data/DualGraphs/V", num2str(N));
    mkdir(subdir)
    num_graphs = data(i,2);
    curr_ids = GraphIDs(id_ptr:id_ptr+num_graphs-1);
    printf("working on dual graphs with %d vertices\n", N)
    printf("There are %d such graphs\n",num_graphs)
  
    graph_file = sprintf('../data/DualAdj/V%dAdjDG', N);
    matrices = load(graph_file);
    
    for j = 1 : size(matrices, 1)/N
        graph_id = curr_ids(j){1};
        AdjM = matrices((j-1)*N+1:j*N, :);
        
        opath = strcat(subdir,"/",graph_id,".txt");
        disp(opath);
        disp(graph_id);
        file_id = fopen(opath, 'w');
        fdisp(file_id,graph_id);
        fdisp(file_id,AdjM);
        fclose(file_id);
    end
    id_ptr = id_ptr+num_graphs;
end