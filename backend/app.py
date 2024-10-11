from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from backend.methods.linalgebra_methods import transform_vector
from backend.methods.numethods_methods import bisection_method, newtonraphson_method

app = Flask(__name__)
CORS(app)

@app.route('/transform', methods=['POST'])
def handle_transform():
    data = request.json
    
    # Extract matrix and vector from request
    matrix = np.array([
        [data['matrix']['a11'], data['matrix']['a12']],
        [data['matrix']['a21'], data['matrix']['a22']]
    ])
    vector = np.array([data['vector']['x'], data['vector']['y']])
    
    # Perform transformation
    transformed_vector = transform_vector(matrix, vector)
    
    return jsonify({
        'transformed_vector': {
            'x': float(transformed_vector[0]),
            'y': float(transformed_vector[1])
        }
    })

@app.route('/bisection', methods=['POST'])
def handle_bisection():
    data = request.json
    equation_str = data['equation']
    a = float(data['interval']['a'])
    b = float(data['interval']['b'])
    
    root, plot_points = bisection_method(equation_str, a, b)
    
    return jsonify({
        'root': float(root) if root is not None else None,
        'plot_points': plot_points
    })

if __name__ == '__main__':
    app.run(debug=True)