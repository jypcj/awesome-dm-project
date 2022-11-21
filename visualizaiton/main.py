import json
import os
from collections import defaultdict

from plotly.subplots import make_subplots
import plotly.graph_objects as go


file_path = [
    './Our_Result/TENT-result_dblp.json',
    './Our_Result/TENT-result_Amazon-eletronics.json',
    './Our_Result/TENT-result_cora-full.json',
    './Our_Result/TENT-result_OGBN-arxiv.json'
]


def parse_json(file_path):
    matrix_data = defaultdict(list)
    with open(file_path, 'r') as file:
        print(os.path.basename(file_path))
        loaded_data = json.load(file)
    for k, v in loaded_data.items():
        k_split = k.split(' ')
        # extract the mean value of N-way k-shot's results
        if len(k_split) == 2 and k_split[1].split('-')[1] == 'shot':
            print(k, round(v[0] * 100, 2))
            # keep 2 decimal places
            matrix_data[k_split[0]].append(round(v[0] * 100, 2))
    print()
    return dict(matrix_data)


# the position of colorbars
# first row, second row
colorbar_pos = [(0.45, 0.81), (1.00, 0.81), (0.45, 0.19), (1.00, 0.19)]


def visualize(data, index, N, K, key_str):
    data_list = []
    print('data:', data)
    print(key_str[0])
    for i in range(len(data[key_str[0]])):

        row = []
        # Use the data with different way and same shot
        for key in key_str:
            row.append(data[key][i])
        # Add the new row data at the beginning of the data_list
        data_list.insert(0, row)
    # String number displayed at the center of cells
    string_data_list = [[j for j in i] for i in data_list]

    # Generate heatmap obj
    fig = go.Heatmap(
        z=data_list,
        x=N,
        y=K,
        text=string_data_list,
        texttemplate="%{text}",
        textfont={"size": 20},
        colorbar=dict(x=colorbar_pos[index][0], y=colorbar_pos[index][1], thickness=25, len=0.38, nticks=8),
        colorscale='Burg',
        reversescale=True
    )
    return fig


if __name__ == '__main__':
    data = []
    # Parse each json file
    for fp in file_path:
        matrix_data = parse_json(fp)
        data.append(matrix_data)

    # Generate subplots
    subplot_titles = ['DBLP', 'Amazon-E', 'Cora-full', 'OGBN-arix']
    fig = make_subplots(rows=2, cols=2, subplot_titles=subplot_titles)


    N = ['3', '5', '10', '15']
    K = ['5', '4', '3']
    key_str = [num + '-way' for num in N]


    index = 0

    # Number of row and column starts at 1
    for i in range(1, 3):
        for j in range(1, 3):
            print(i, j)
            print(data)
            if index == 3:
                subfig = visualize(data[index], index, ['5', '10'], ['5', '3'], ['5-way', '10-way'])
            else:
                subfig = visualize(data[index], index, N, K, key_str)
            fig.add_trace(subfig, row=i, col=j)
            fig.update_xaxes(title_text='Value of Nt', row=i, col=j, title_font_size=16)
            fig.update_yaxes(title_text='Value of Kt', row=i, col=j, title_font_size=16)
            index += 1

    fig.update_layout(title_text='Results of TENT with different Nt and Kt', title_x=0.5, title_font_size=28)

    fig.show()
