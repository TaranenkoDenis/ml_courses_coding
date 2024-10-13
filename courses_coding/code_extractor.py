import nbformat

def extract_code_from_ipynb(ipynb_file, output_file):
    print('1')
    with open(ipynb_file, 'r', encoding='utf-8') as file:
        notebook = nbformat.read(file, as_version=4)
    print('1')
    code_cells = [cell['source'] for cell in notebook['cells'] if cell['cell_type'] == 'code']
    print('1')
    with open(output_file, 'w', encoding='utf-8') as file:
        for i, code in enumerate(code_cells, 1):
            file.write(code)
            file.write('\n\n')
    print('1')

# Replace 'notebook.ipynb' and 'output.py' with your file names
extract_code_from_ipynb('courses_coding/8_deep_learning/convolutional_neural_network.ipynb', 'output.py')