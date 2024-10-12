import React, { useState, useCallback } from 'react';
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";

const VectorFieldVisualization = ({ F_x, F_y, width = 300, height = 300, gridSize = 10 }) => {
  const scale = 20;
  const arrowSize = 5;

  const drawVector = (x, y) => {
    const startX = x * (width / gridSize);
    const startY = height - y * (height / gridSize);
    const endX = startX + F_x(x, y) * scale;
    const endY = startY - F_y(x, y) * scale;

    const angle = Math.atan2(startY - endY, endX - startX);

    return (
      <g key={`${x}-${y}`}>
        <line
          x1={startX}
          y1={startY}
          x2={endX}
          y2={endY}
          stroke="blue"
          strokeWidth="1"
        />
        <polygon
          points={`0,-${arrowSize} ${arrowSize},0 0,${arrowSize}`}
          fill="blue"
          transform={`translate(${endX},${endY}) rotate(${angle * (180 / Math.PI)})`}
        />
      </g>
    );
  };

  const vectors = [];
  for (let x = 0; x <= gridSize; x++) {
    for (let y = 0; y <= gridSize; y++) {
      vectors.push(drawVector(x, y));
    }
  }

  return (
    <svg width={width} height={height}>
      {vectors}
    </svg>
  );
};

const VectorFieldAnalysis = () => {
  const [F_x, setF_x] = useState('y');
  const [F_y, setF_y] = useState('x');
  const [x, setX] = useState(1);
  const [y, setY] = useState(1);
  const [divergenceResult, setDivergenceResult] = useState(null);
  const [curlResult, setCurlResult] = useState(null);

  const calculateDivergence = useCallback((F_x, F_y, x, y, h = 1e-6) => {
    const dFx_dx = (eval(`(x, y) => ${F_x}`)(x + h, y) - eval(`(x, y) => ${F_x}`)(x - h, y)) / (2 * h);
    const dFy_dy = (eval(`(x, y) => ${F_y}`)(x, y + h) - eval(`(x, y) => ${F_y}`)(x, y - h)) / (2 * h);
    return dFx_dx + dFy_dy;
  }, []);

  const calculateCurl = useCallback((F_x, F_y, x, y, h = 1e-6) => {
    const dFy_dx = (eval(`(x, y) => ${F_y}`)(x + h, y) - eval(`(x, y) => ${F_y}`)(x - h, y)) / (2 * h);
    const dFx_dy = (eval(`(x, y) => ${F_x}`)(x, y + h) - eval(`(x, y) => ${F_x}`)(x, y - h)) / (2 * h);
    return dFy_dx - dFx_dy;
  }, []);

  const handleCalculate = () => {
    try {
      const divergence = calculateDivergence(F_x, F_y, parseFloat(x), parseFloat(y));
      const curl = calculateCurl(F_x, F_y, parseFloat(x), parseFloat(y));
      setDivergenceResult(divergence);
      setCurlResult(curl);
    } catch (error) {
      console.error('Error in calculation:', error);
      setDivergenceResult('Error');
      setCurlResult('Error');
    }
  };

  return (
    <Card>
      <CardHeader>2D Vector Field Analysis</CardHeader>
      <CardContent>
        <div className="space-y-4">
          <div>
            <Label htmlFor="F_x">F_x(x, y) = </Label>
            <Input
              id="F_x"
              value={F_x}
              onChange={(e) => setF_x(e.target.value)}
              placeholder="Enter F_x function"
            />
          </div>
          <div>
            <Label htmlFor="F_y">F_y(x, y) = </Label>
            <Input
              id="F_y"
              value={F_y}
              onChange={(e) => setF_y(e.target.value)}
              placeholder="Enter F_y function"
            />
          </div>
          <div className="grid grid-cols-2 gap-2">
            <div>
              <Label htmlFor="x">x = </Label>
              <Input
                id="x"
                type="number"
                value={x}
                onChange={(e) => setX(e.target.value)}
              />
            </div>
            <div>
              <Label htmlFor="y">y = </Label>
              <Input
                id="y"
                type="number"
                value={y}
                onChange={(e) => setY(e.target.value)}
              />
            </div>
          </div>
          <Button onClick={handleCalculate}>Calculate</Button>
          
          <div className="mt-4">
            <h3 className="font-semibold mb-2">Results:</h3>
            <p>Divergence: {divergenceResult !== null ? divergenceResult.toFixed(4) : 'N/A'}</p>
            <p>Curl (z-component): {curlResult !== null ? curlResult.toFixed(4) : 'N/A'}</p>
          </div>
          
          <div className="mt-4">
            <h3 className="font-semibold mb-2">Vector Field Visualization:</h3>
            <VectorFieldVisualization
              F_x={(x, y) => eval(`(x, y) => ${F_x}`)(x, y)}
              F_y={(x, y) => eval(`(x, y) => ${F_y}`)(x, y)}
            />
          </div>
        </div>
      </CardContent>
    </Card>
  );
};

export default VectorFieldAnalysis;