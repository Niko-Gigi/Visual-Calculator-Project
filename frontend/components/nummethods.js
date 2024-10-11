import React, { useState } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, ResponsiveContainer } from 'recharts';
import { Card, CardContent, CardHeader } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group"
import { Label } from "@/components/ui/label"

export default function NumericalMethods() {
  const [method, setMethod] = useState('bisection');
  const [equation, setEquation] = useState('x^2 - 4');
  const [interval, setInterval] = useState({ a: 0, b: 3 });
  const [initialGuess, setInitialGuess] = useState(2);
  const [result, setResult] = useState(null);
  const [plotData, setPlotData] = useState([]);
  const [iterations, setIterations] = useState([]);

  const handleCalculation = async () => {
    try {
      const endpoint = method === 'bisection' ? '/bisection' : '/newton';
      const body = method === 'bisection' 
        ? { equation, interval }
        : { equation, x0: initialGuess };

      const response = await fetch(`http://localhost:5000${endpoint}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body)
      });
      const data = await response.json();
      setResult(data.root);
      setPlotData(data.plot_points);
      if (data.iterations) setIterations(data.iterations);
    } catch (error) {
      console.error('Error in calculation:', error);
    }
  };

  return (
    <Card>
      <CardHeader>Numerical Methods</CardHeader>
      <CardContent>
        <div className="space-y-4">
          <RadioGroup
            defaultValue="bisection"
            onValueChange={setMethod}
            className="flex space-x-4"
          >
            <div className="flex items-center space-x-2">
              <RadioGroupItem value="bisection" id="bisection" />
              <Label htmlFor="bisection">Bisection Method</Label>
            </div>
            <div className="flex items-center space-x-2">
              <RadioGroupItem value="newton" id="newton" />
              <Label htmlFor="newton">Newton-Raphson Method</Label>
            </div>
          </RadioGroup>

          <div>
            <Label htmlFor="equation">Equation</Label>
            <Input 
              id="equation"
              value={equation} 
              onChange={e => setEquation(e.target.value)}
              placeholder="Enter equation (e.g., x^2 - 4)"
            />
          </div>

          {method === 'bisection' ? (
            <div className="grid grid-cols-2 gap-4">
              <div>
                <Label htmlFor="a">Left Endpoint (a)</Label>
                <Input 
                  id="a"
                  type="number" 
                  value={interval.a} 
                  onChange={e => setInterval({...interval, a: parseFloat(e.target.value)})}
                />
              </div>
              <div>
                <Label htmlFor="b">Right Endpoint (b)</Label>
                <Input 
                  id="b"
                  type="number" 
                  value={interval.b} 
                  onChange={e => setInterval({...interval, b: parseFloat(e.target.value)})}
                />
              </div>
            </div>
          ) : (
            <div>
              <Label htmlFor="x0">Initial Guess (x₀)</Label>
              <Input 
                id="x0"
                type="number" 
                value={initialGuess} 
                onChange={e => setInitialGuess(parseFloat(e.target.value))}
              />
            </div>
          )}

          <Button onClick={handleCalculation}>Find Root</Button>
          
          {result && (
            <div className="mt-4">
              <h3 className="font-semibold mb-2">Result:</h3>
              <p>Root found at x = {result}</p>
              {iterations.length > 0 && (
                <div className="mt-2">
                  <h4 className="font-semibold">Iterations:</h4>
                  <ul className="list-decimal list-inside">
                    {iterations.map((x, i) => (
                      <li key={i}>x_{i} = {x.toFixed(6)}</li>
                    ))}
                  </ul>
                </div>
              )}
              {plotData.length > 0 && (
                <div className="h-64 mt-4">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={plotData}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="x" />
                      <YAxis />
                      <Line type="monotone" dataKey="y" stroke="#8884d8" />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              )}
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
}