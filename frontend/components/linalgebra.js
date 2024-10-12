import React, { useState } from 'react';
import { Card, CardContent, CardHeader } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"

const VectorVisualization = ({ originalVector, transformedVector }) => {
  const scale = 20; // Scale factor for better visualization
  const center = 100; // Center point of the SVG

  const drawVector = (vector, color) => {
    const x = center + vector.x * scale;
    const y = center - vector.y * scale; // Subtract because SVG y-axis is inverted
    return (
      <g>
        <line x1={center} y1={center} x2={x} y2={y} stroke={color} strokeWidth="2" />
        <circle cx={x} cy={y} r="3" fill={color} />
      </g>
    );
  };

  return (
    <svg width="200" height="200" viewBox="0 0 200 200">
      {/* Coordinate axes */}
      <line x1="0" y1={center} x2="200" y2={center} stroke="black" strokeWidth="1" />
      <line x1={center} y1="0" x2={center} y2="200" stroke="black" strokeWidth="1" />
      
      {/* Original vector */}
      {drawVector(originalVector, "blue")}
      
      {/* Transformed vector */}
      {transformedVector && drawVector(transformedVector, "red")}
    </svg>
  );
};

export default function LinearAlgebra() {
  const [matrix, setMatrix] = useState({ a11: 1, a12: 0, a21: 0, a22: 1 });
  const [vector, setVector] = useState({ x: 1, y: 1 });
  const [result, setResult] = useState(null);

  const handleTransform = async () => {
    try {
      const response = await fetch('http://localhost:5000/transform', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ matrix, vector })
      });
      const data = await response.json();
      setResult(data.transformed_vector);
    } catch (error) {
      console.error('Error in transformation:', error);
    }
  };

  return (
    <Card>
      <CardHeader>Linear Algebra - 2D Vector Transformation</CardHeader>
      <CardContent>
        <div className="space-y-4">
          <div>
            <h3 className="font-semibold mb-2">Transformation Matrix</h3>
            <div className="grid grid-cols-2 gap-2">
              <Input
                type="number"
                value={matrix.a11}
                onChange={e => setMatrix({...matrix, a11: parseFloat(e.target.value)})}
                placeholder="a11"
              />
              <Input
                type="number"
                value={matrix.a12}
                onChange={e => setMatrix({...matrix, a12: parseFloat(e.target.value)})}
                placeholder="a12"
              />
              <Input
                type="number"
                value={matrix.a21}
                onChange={e => setMatrix({...matrix, a21: parseFloat(e.target.value)})}
                placeholder="a21"
              />
              <Input
                type="number"
                value={matrix.a22}
                onChange={e => setMatrix({...matrix, a22: parseFloat(e.target.value)})}
                placeholder="a22"
              />
            </div>
          </div>
          <div>
            <h3 className="font-semibold mb-2">Vector</h3>
            <div className="grid grid-cols-2 gap-2">
              <Input
                type="number"
                value={vector.x}
                onChange={e => setVector({...vector, x: parseFloat(e.target.value)})}
                placeholder="x"
              />
              <Input
                type="number"
                value={vector.y}
                onChange={e => setVector({...vector, y: parseFloat(e.target.value)})}
                placeholder="y"
              />
            </div>
          </div>
          <Button onClick={handleTransform}>Transform</Button>
         
          <div className="mt-4">
            <h3 className="font-semibold mb-2">Visualization:</h3>
            <VectorVisualization originalVector={vector} transformedVector={result} />
            <div className="text-sm mt-2">
              <p>Blue: Original vector</p>
              <p>Red: Transformed vector</p>
            </div>
          </div>

          {result && (
            <div className="mt-4">
              <h3 className="font-semibold mb-2">Result:</h3>
              <p>Transformed vector: ({result.x.toFixed(2)}, {result.y.toFixed(2)})</p>
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
}