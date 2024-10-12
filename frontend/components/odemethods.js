import React, { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, ResponsiveContainer } from 'recharts';
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";

const CobwebPlot = () => {
  const [r, setR] = useState(2.8);
  const [x0, setX0] = useState(0.2);
  const [nmax, setNmax] = useState(40);
  const [plotData, setPlotData] = useState(null);

  const fetchData = async () => {
    try {
      const response = await fetch('http://localhost:5000/cobweb', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ r, x0, nmax }),
      });
      const data = await response.json();
      setPlotData(data);
    } catch (error) {
      console.error('Error fetching data:', error);
    }
  };

  useEffect(() => {
    fetchData();
  }, []);

  const handleCalculate = () => {
    fetchData();
  };

  const renderPlot = () => {
    if (!plotData) return null;

    const { x, fx, px, py } = plotData;
    const plotPoints = x.map((xi, i) => ({ x: xi, fx: fx[i], y: xi }));
    const cobwebPoints = px.map((pxi, i) => ({ x: pxi, y: py[i] }));

    return (
      <ResponsiveContainer width="100%" height={400}>
        <LineChart margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="x" type="number" domain={[0, 1]} />
          <YAxis type="number" domain={[0, 1]} />
          <Line data={plotPoints} type="monotone" dataKey="fx" stroke="#8884d8" dot={false} />
          <Line data={plotPoints} type="monotone" dataKey="y" stroke="#82ca9d" dot={false} />
          <Line data={cobwebPoints} type="monotone" dataKey="y" stroke="#ff7300" dot={false} />
        </LineChart>
      </ResponsiveContainer>
    );
  };

  return (
    <Card>
      <CardHeader>Cobweb Plot</CardHeader>
      <CardContent>
        <div className="space-y-4">
          <div>
            <Label htmlFor="r">r value:</Label>
            <Input
              id="r"
              type="number"
              value={r}
              onChange={(e) => setR(parseFloat(e.target.value))}
              step="0.1"
            />
          </div>
          <div>
            <Label htmlFor="x0">Initial x value (x0):</Label>
            <Input
              id="x0"
              type="number"
              value={x0}
              onChange={(e) => setX0(parseFloat(e.target.value))}
              step="0.1"
            />
          </div>
          <div>
            <Label htmlFor="nmax">Number of iterations:</Label>
            <Input
              id="nmax"
              type="number"
              value={nmax}
              onChange={(e) => setNmax(parseInt(e.target.value))}
            />
          </div>
          <Button onClick={handleCalculate}>Calculate</Button>
          {renderPlot()}
        </div>
      </CardContent>
    </Card>
  );
};

export default CobwebPlot;