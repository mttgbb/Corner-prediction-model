## This script contains functions for generating various graphs in both dictionary and adjacency matrix forms

# Function for enumerating vertices with a tuple (useful for lattices)
function vertex_indices(dims::Dims{2}; unitcellN::Int = 1)
		return [(c1,c2,c3) for c1 in 0:dims[1]-1, c2 in 0:dims[2]-1, c3 in 0:unitcellN-1]
	end

# Function to generate custom graphs as a dictionary
function J_dict(dims::Dims{2}, vertices::Array{Tuple{Int, Int, Int}}, J_set::Vector{Float64}, Neighbour_set::Vector{Vector{Tuple{Int, Int}}}; unitcellN::Int = 1, manual_vertices::Vector{Tuple{Int, Int, Int}} = Vector{Tuple{Int, Int, Int}}(), manual_J_set::Vector{Vector{Float64}} = Vector{Vector{Float64}}(), manual_Neighbour_set::Vector{Vector{Vector{Tuple{Int, Int}}}} = Vector{Vector{Vector{Tuple{Int, Int}}}}()) 
    J = Dict{Tuple{Int, Int, Int}, Set{Edge}}()
    for vertex in vertices                  
        edge_set = Set{Edge}()
        manual_neighbours = Set{Tuple{Int, Int, Int}}()
        if vertex in manual_vertices
			manual_index = findfirst(x -> x == vertex, manual_vertices)
			sub_manual_J_set = manual_J_set[manual_index]
            for k in 1:length(sub_manual_J_set)
                for shift in manual_Neighbour_set[manual_index][k]
					if unitcellN == 1
						neighbour_index = (mod(vertex[1]+shift[1], dims[1]), mod(vertex[2]+shift[2], dims[2]), 0)
					elseif unitcellN == 2
						if vertex[3] == 1
							neighbour_index = (mod(vertex[1]+shift[1], dims[1]), mod(vertex[2]+shift[2], dims[2]), 0)
						else
							neighbour_index = (mod(vertex[1]-shift[1], dims[1]), mod(vertex[2]-shift[2], dims[2]), 1)
						end
					end							
					if neighbour_index in vertices
                       	push!(manual_neighbours, neighbour_index)
                       	push!(edge_set, Edge(neighbour_index, 
						sub_manual_J_set[k]))
					end
                end
            end
        end
        for k in 1:length(J_set)
            for shift in Neighbour_set[k]
            	if unitcellN == 1
					neighbour_index = (mod(vertex[1]+shift[1], dims[1]), mod(vertex[2]+shift[2], dims[2]), 0)
				elseif unitcellN == 2
					if vertex[3] == 1
						neighbour_index = (mod(vertex[1]+shift[1], dims[1]), mod(vertex[2]+shift[2], dims[2]), 0)
					else
						neighbour_index = (mod(vertex[1]-shift[1], dims[1]), mod(vertex[2]-shift[2], dims[2]), 1)
					end
				end	
                if neighbour_index in vertices && !(neighbour_index in 
				manual_neighbours)
                   	push!(edge_set, Edge(neighbour_index, J_set[k]))
                end
            end
        end
        J[vertex] = edge_set
	end
    return J
end

# Function for external magnetic field
function h_dict(vertices::Array{Tuple{Int, Int, Int}}, h::Float64)
    h = Dict(i => h for i in vertices)
    return h
end

# Functions for converting between adjacency matrices and dictionary representation of graphs
function adj_matrix_to_J_dict(vertices::Array{Tuple{Int, Int, Int}}, adj_matrix::Matrix{Float64})
	J = Dict{Tuple{Int, Int, Int}, Set{Edge}}()
	enum_vertices = enumerate(vertices)
	for (i, src) in enum_vertices
		edge_set = Set{Edge}()
		for (j, dst) in enum_vertices
			weight_ij = adj_matrix[i, j]
			if weight_ij != 0.0
				push!(edge_set, Edge(dst, weight_ij))	
			end
		end
		J[src] = edge_set
	end
    return J
end

function J_dict_to_adj_matrix(vertices::Array{Tuple{Int, Int, Int}}, J_dict::Dict{Tuple{Int, Int, Int}, Set{Edge}})
	vertices_vec = vec(vertices)
	adj_matrix = zeros(Float64, length(vertices), length(vertices))
	for i in 1:length(vertices_vec)
		for edge in J_dict[vertices_vec[i]]
			dst = getproperty(edge, :dst)
			j = findfirst(x -> x == dst, vertices_vec)
			adj_matrix[i, j] = getproperty(edge, :weight)
		end
	end
	return adj_matrix
end
	
# Functions for generating lattices with higher order nearest neighbour couplings
function find_distance(d::Int, e1, e2; tol = 1e-6)
    distances = []
    for x in -d:d
        for y in -d:d
            if x != 0 || y != 0
				coord = x*e1 + y*e2
                dist = norm(coord)
                push!(distances, dist)
            end
        end
    end
    
    function unique_with_tol(distances, tol)
        sorted_distances = sort(distances)
        unique_distances = [sorted_distances[1]]
        
        for i in 2:length(sorted_distances)
            if abs(sorted_distances[i] - sorted_distances[i-1]) > tol
                push!(unique_distances, sorted_distances[i])
            end
        end
        return unique_distances
    end

    unique_distances = unique_with_tol(distances, tol)
    
    if d > length(unique_distances)
        error("d not in bounds")
    end
    return unique_distances[d]
end

function generate_neighbors(d::Int, e1, e2; tol = 1e-6)
	dist_thres = find_distance(d, e1, e2; tol = tol)
    neighbors = Vector{Dims{2}}()
    for x in -d:d
        for y in -d:d
            if x != 0 || y != 0
               coord = x*e1 + y*e2
                dist = norm(coord)
                if abs(dist - dist_thres) < tol
                	push!(neighbors, (x, y))
                end
            end
        end
    end
    return neighbors
end

# Functions for generating various periodic BC cubic lattices
function adjacency_matrix_1d_chain_pbc(n::Int)
    A = zeros(Float64, n, n)
    
    for i in 1:n
        neighbor_next = mod1(i + 1, n)
        A[i, neighbor_next] = 1.0
        A[neighbor_next, i] = 1.0
        
        neighbor_prev = mod1(i - 1, n)
        A[i, neighbor_prev] = 1.0
        A[neighbor_prev, i] = 1.0
    end
    
    return A
end
	
function adjacency_matrix_2d_lattice_pbc(n::Int)
    num_vertices = n^2
    A = zeros(Float64, num_vertices, num_vertices)
    
    function index(i::Int, j::Int, n::Int)
        return (i - 1) * n + j
    end
    
    for i in 1:n
        for j in 1:n
            current = index(i, j, n)
            neighbor_up = index(mod1(i - 1, n), j, n)
            neighbor_down = index(mod1(i + 1, n), j, n)
            neighbor_left = index(i, mod1(j - 1, n), n)
            neighbor_right = index(i, mod1(j + 1, n), n)
            
            A[current, neighbor_up] = 1.0
            A[neighbor_up, current] = 1.0
            A[current, neighbor_down] = 1.0
            A[neighbor_down, current] = 1.0
            A[current, neighbor_left] = 1.0
            A[neighbor_left, current] = 1.0
            A[current, neighbor_right] = 1.0
            A[neighbor_right, current] = 1.0
        end
    end
    
    return A
end


function adjacency_matrix_3d_lattice_pbc(n::Int, m::Int, p::Int)
    num_vertices = n * m * p
    A = zeros(Float64, num_vertices, num_vertices)
    
    function index(i::Int, j::Int, k::Int, n::Int, m::Int)
        return (i - 1) * m * p + (j - 1) * p + k
    end
    
    for i in 1:n
        for j in 1:m
            for k in 1:p
                current = index(i, j, k, n, m)
                
                neighbor_front = index(mod1(i - 1, n), j, k, n, m)
                neighbor_back = index(mod1(i + 1, n), j, k, n, m)
                neighbor_left = index(i, mod1(j - 1, m), k, n, m)
                neighbor_right = index(i, mod1(j + 1, m), k, n, m)
                neighbor_down = index(i, j, mod1(k - 1, p), n, m)
                neighbor_up = index(i, j, mod1(k + 1, p), n, m)
                
                A[current, neighbor_front] = 1.0
                A[neighbor_front, current] = 1.0
                A[current, neighbor_back] = 1.0
                A[neighbor_back, current] = 1.0
                A[current, neighbor_left] = 1.0
                A[neighbor_left, current] = 1.0
                A[current, neighbor_right] = 1.0
                A[neighbor_right, current] = 1.0
                A[current, neighbor_down] = 1.0
                A[neighbor_down, current] = 1.0
                A[current, neighbor_up] = 1.0
                A[neighbor_up, current] = 1.0
            end
        end
    end
    
    return A
end	

# Functions for generating 1D, 2D free BC cubic lattices
function adjacency_matrix_1d_chain_free(n::Int)
    adj_matrix = zeros(Float64, n, n)   
	
    for i in 2:n-1
        adj_matrix[i, i-1] = 1.0 
        adj_matrix[i, i+1] = 1.0  
    end
	
	adj_matrix[1, 2] = 1.0
	adj_matrix[n, n-1] = 1.0  
	
    return adj_matrix
end

function adjacency_matrix_2d_square_lattice_free(N::Int)
    num_nodes = N^2
    A = zeros(Float64, num_nodes, num_nodes)

    index = (i, j) -> (i - 1) * N + j
	
    for i in 1:N
        for j in 1:N
            node = index(i, j)

            if j < N
                right = index(i, j + 1)
                A[node, right] = 1.0
                A[right, node] = 1.0
            end

            if i < N
                bottom = index(i + 1, j)
                A[node, bottom] = 1.0
                A[bottom, node] = 1.0
            end
        end
    end

    return A
end

# Function for generating n-dimensional hypercube (multiply adjacency matrix by 2 to get periodic boundary conditions)
function adjacency_matrix_hypercube(n::Int)
    num_vertices = 2^n
    A = zeros(Float64, num_vertices, num_vertices)

	function hamming_distance(x::Int, y::Int)
    	return count_ones(x ⊻ y)
	end
	
    for i in 1:num_vertices
        for j in i+1:num_vertices
            if hamming_distance(i-1, j-1) == 1
                A[i, j] = 1.0
                A[j, i] = 1.0
            end
        end
    end

    return A
end

# Function for generating star graph
function adjacency_matrix_star(n::Int)
    A = zeros(Float64, n, n)

    for i in 2:n
        A[1, i] = 1.0
        A[i, 1] = 1.0
    end

    return A
end

# Function for generating layered star graph
function adjacency_matrix_layered_star(n_layers::Int, layer_size::Int)
    n_nodes = 1 + n_layers * layer_size
    A = zeros(Float64, n_nodes, n_nodes)

    for i in 2:(1 + layer_size)
        A[1, i] = 1.0
        A[i, 1] = 1.0
    end

    for layer in 1:(n_layers-1)
        start_idx_curr = 2 + (layer - 1) * layer_size
        start_idx_next = 2 + layer * layer_size

        for i in 0:(layer_size-1)
            A[start_idx_curr + i, start_idx_next + i] = 1.0
            A[start_idx_next + i, start_idx_curr + i] = 1.0
        end
    end

    return A
end

# Function for generating complete graph
function adjacency_matrix_complete(n::Int)
    A = zeros(Float64, n, n)

    for i in 1:n
        for j in i+1:n
            A[i, j] = 1.0
            A[j, i] = 1.0
        end
    end

    return A
end

# Functions for generating wheel graphs and variants of wheel graphs
function adjacency_matrix_wheel(n::Int)
    A = zeros(Float64, n, n)

    for i in 2:n
        A[1, i] = 1.0
        A[i, 1] = 1.0
    end

    for i in 2:n-1
        A[i, i+1] = 1.0
        A[i+1, i] = 1.0
    end
    A[2, n] = 1.0
    A[n, 2] = 1.0

    return A
end
	
function adjacency_matrix_ntuple_wheel(n::Int, layer_size::Int)
    total_nodes = 1 + n * layer_size 
    A = zeros(Float64, total_nodes, total_nodes)

    for i in 2:total_nodes
        A[1, i] = 1.0
        A[i, 1] = 1.0
    end

    for i in 1:n
        start_idx = 1 + (i - 1) * layer_size
        for j in 0:layer_size-2
            A[start_idx + j, start_idx + j + 1] = 1.0
            A[start_idx + j + 1, start_idx + j] = 1.0
        end
        A[start_idx, start_idx + layer_size - 1] = 1.0
        A[start_idx + layer_size - 1, start_idx] = 1.0
    end

    return A
end

function adjacency_matrix_layered_wheel(n_layers::Int, layer_size::Int)
    n_nodes = 1 + n_layers * layer_size
    A = zeros(Float64, n_nodes, n_nodes)

    for i in 2:(1 + layer_size)
        A[1, i] = 1.0
        A[i, 1] = 1.0
    end

    for layer in 0:(n_layers-1)
        start_idx = 2 + layer * layer_size
        end_idx = start_idx + layer_size - 1

        for i in start_idx:(end_idx-1)
            A[i, i+1] = 1.0
            A[i+1, i] = 1.0
        end
        A[end_idx, start_idx] = 1.0
        A[start_idx, end_idx] = 1.0
    end

    for layer in 0:(n_layers-2)
        start_idx_curr = 2 + layer * layer_size
        start_idx_next = 2 + (layer + 1) * layer_size

        for i in 0:(layer_size-1)
            A[start_idx_curr + i, start_idx_next + i] = 1.0
            A[start_idx_next + i, start_idx_curr + i] = 1.0
        end
    end

    return A
end

# Functions for generating barbell graphs and variants of barbell graphs
function adjacency_matrix_barbell(n1::Int, n2::Int)
    total_vertices = 2 * n1 + n2
    A = zeros(Float64, total_vertices, total_vertices)

    for i in 1:n1
        for j in i+1:n1
            A[i, j] = 1.0
            A[j, i] = 1.0
        end
    end

    for i in n1+n2+1:total_vertices
        for j in i+1:total_vertices
            A[i, j] = 1.0
            A[j, i] = 1.0
        end
    end

    A[n1, n1+1] = 1.0
    A[n1+1, n1] = 1.0

    for i in n1+1:n1+n2-1
        A[i, i+1] = 1.0
        A[i+1, i] = 1.0
    end

    A[n1+n2, n1+n2+1] = 1.0
    A[n1+n2+1, n1+n2] = 1.0

    return A
end

function adjacency_matrix_skewed_barbell(n1::Int, n2::Int, n3::Int)
    total_vertices = n1 + n2 + n3
    A = zeros(Float64, total_vertices, total_vertices)

    for i in 1:n1
        for j in i+1:n1
            A[i, j] = 1.0
            A[j, i] = 1.0
        end
    end

    for i in n1+n2+1:total_vertices
        for j in i+1:total_vertices
            A[i, j] = 1.0
            A[j, i] = 1.0
        end
    end

    A[n1, n1+1] = 1.0
    A[n1+1, n1] = 1.0

    for i in n1+1:n1+n2-1
        A[i, i+1] = 1.0
        A[i+1, i] = 1.0
    end

    A[n1+n2, n1+n2+1] = 1.0
    A[n1+n2+1, n1+n2] = 1.0

    return A
end

function adjacency_matrix_barbell_2dlink(n1_left::Int, n1_right::Int, n2_width::Int, n2_height::Int)
    total_vertices = n1_left + n1_right + n2_width * n2_height
    A = zeros(Float64, total_vertices, total_vertices)

    for i in 1:n1_left
        for j in i+1:n1_left
            A[i, j] = 1.0
            A[j, i] = 1.0
        end
    end

    for i in n1_left + n2_width * n2_height + 1:total_vertices
        for j in i+1:total_vertices
            A[i, j] = 1.0
            A[j, i] = 1.0
        end
    end

    start_link_index = n1_left + 1
    for row in 0:n2_height-1
        for col in 0:n2_width-1
            current_vertex = start_link_index + row * n2_width + col

            if col < n2_width - 1
                right_neighbor = current_vertex + 1
                A[current_vertex, right_neighbor] = 1.0
                A[right_neighbor, current_vertex] = 1.0
            end

            if row < n2_height - 1
                bottom_neighbor = current_vertex + n2_width
                A[current_vertex, bottom_neighbor] = 1.0
                A[bottom_neighbor, current_vertex] = 1.0
            end

			if col == 1
				A[current_vertex, n1_left - row] = 1.0
				A[n1_left - row, current_vertex] = 1.0
			elseif col == n2_width
				A[current_vertex, n1_left + n2_width * n2_height + 1 + row] = 1.0
				A[n1_left + n2_width * n2_height + 1 + row, current_vertex] = 1.0
			end
        end
    end

    return A
end

function adjacency_matrix_barbell_multiplelinks(n1_left::Int, n1_right::Int, n2_width::Int, n2_height::Int)
    total_vertices = n1_left + n1_right + n2_width * n2_height
    A = zeros(Float64, total_vertices, total_vertices)

    for i in 1:n1_left
        for j in i+1:n1_left
            A[i, j] = 1.0
            A[j, i] = 1.0
        end
    end

    for i in n1_left + n2_width * n2_height + 1:total_vertices
        for j in i+1:total_vertices
            A[i, j] = 1.0
            A[j, i] = 1.0
        end
    end

    start_link_index = n1_left + 1
    for row in 0:n2_height-1
        for col in 0:n2_width-1
            current_vertex = start_link_index + row * n2_width + col

            if col < n2_width - 1
                right_neighbor = current_vertex + 1
                A[current_vertex, right_neighbor] = 1.0
                A[right_neighbor, current_vertex] = 1.0
            end

			if col == 1
				A[current_vertex, n1_left - row] = 1.0
				A[n1_left - row, current_vertex] = 1.0
			elseif col == n2_width
				A[current_vertex, n1_left + n2_width * n2_height + 1 + row] = 1.0
				A[n1_left + n2_width * n2_height + 1 + row, current_vertex] = 1.0
			end
        end
    end

    return A
end
