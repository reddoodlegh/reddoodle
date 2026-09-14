import torch


def rowswap(matrix, source_row_index, target_row_index):
    source_row = matrix[source_row_index, :].clone() # preserve the rows
    target_row = matrix[target_row_index, :].clone()
    # Swap
    matrix[source_row_index, :] = target_row
    matrix[target_row_index, :] = source_row
    return matrix


def rowscale(matrix, row_index, scalar):
    matrix[row_index, :] *= scalar
    return matrix


def rowreplacement(matrix, first_row_index, second_row_index, scalar_j, scalar_k):
    # Returns a single row of the first row scaled added to the second row scaled
    first_row_scaled = rowscale(matrix.clone(), first_row_index, scalar_j)
    second_row_scaled = rowscale(matrix.clone(), second_row_index, scalar_k)
    return first_row_scaled[first_row_index, :] + second_row_scaled[second_row_index, :]


def rref(matrix):
    float_matrix = matrix.float().clone()  # Convert to float 
    rows, cols = float_matrix.shape # get number of rows and columns
    pivot_row_index = 0 
    for column_index in range(cols):   # For each column in the matrix
        if pivot_row_index >= rows:
            break
        # Find the pivot row
        pivot_row = None
        for row_index in range(pivot_row_index, rows):
            if float_matrix[row_index, column_index] != 0:
                pivot_row = row_index
                break
        if pivot_row is None:  # No pivot found in this column
            continue  # move to next column

        # Swap the current row with the pivot row
        float_matrix = rowswap(float_matrix, pivot_row_index, pivot_row)

        # Divide the pivot row by the pivot element to make it 1
        pivot_element = float_matrix[pivot_row_index, column_index]
        float_matrix = rowscale(float_matrix, pivot_row_index, 1 / pivot_element)
        
        # Eliminate the current column in other rows
        for row_index in range(rows):
            if row_index != pivot_row_index:
                factor = -float_matrix[row_index, column_index]
                rowreplacement_result = rowreplacement(float_matrix, pivot_row_index, row_index, factor, 1)
                float_matrix[row_index, :] = rowreplacement_result

        pivot_row_index += 1  # Move to the next row for the next pivot
    return float_matrix
