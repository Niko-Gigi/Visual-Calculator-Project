import React, { useState } from 'react';
import { Card, CardContent, CardHeader } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"

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
          
          {result && (
            <div className="mt-4">
              <h3 className="font-semibold mb-2">Result:</h3>
              <p>Transformed vector: ({result.x}, {result.y})</p>
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
}