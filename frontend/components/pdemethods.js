import React, { useState, useEffect, useRef } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, ResponsiveContainer } from 'recharts';
import { Button, Slider } from '@/components/ui/';

const WaveEquationSimulator = () => {
  const [simulationData, setSimulationData] = useState([]);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentStep, setCurrentStep] = useState(0);
  const [waveSpeed, setWaveSpeed] = useState(1);
  const [gridPoints, setGridPoints] = useState(50);
  
  const animationRef = useRef();

  useEffect(() => {
    fetchSimulationData();
  }, [waveSpeed, gridPoints]);

  const fetchSimulationData = async () => {
    try {
      const response = await fetch('/api/simulate', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ waveSpeed, gridPoints }),
      });
      const data = await response.json();
      setSimulationData(data);
      setCurrentStep(0);
    } catch (error) {
      console.error('Error fetching simulation data:', error);
    }
  };

  const animate = () => {
    setCurrentStep((prevStep) => (prevStep + 1) % simulationData.length);
    animationRef.current = requestAnimationFrame(animate);
  };

  useEffect(() => {
    if (isPlaying) {
      animationRef.current = requestAnimationFrame(animate);
    } else {
      cancelAnimationFrame(animationRef.current);
    }
    return () => cancelAnimationFrame(animationRef.current);
  }, [isPlaying]);

  const togglePlayPause = () => {
    setIsPlaying(!isPlaying);
  };

  const handleWaveSpeedChange = (value) => {
    setWaveSpeed(value);
  };

  const handleGridPointsChange = (value) => {
    setGridPoints(value);
  };

  return (
    <div className="p-4">
      <h1 className="text-2xl font-bold mb-4">Wave Equation Simulator</h1>
      <div className="mb-4">
        <Button onClick={togglePlayPause}>
          {isPlaying ? 'Pause' : 'Play'}
        </Button>
      </div>
      <div className="mb-4">
        <label className="block mb-2">Wave Speed: {waveSpeed}</label>
        <Slider
          min={0.1}
          max={2}
          step={0.1}
          value={waveSpeed}
          onValueChange={handleWaveSpeedChange}
        />
      </div>
      <div className="mb-4">
        <label className="block mb-2">Grid Points: {gridPoints}</label>
        <Slider
          min={10}
          max={100}
          step={1}
          value={gridPoints}
          onValueChange={handleGridPointsChange}
        />
      </div>
      <ResponsiveContainer width="100%" height={400}>
        <LineChart data={simulationData[currentStep]}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="x" />
          <YAxis domain={[-2, 2]} />
          <Line type="monotone" dataKey="u" stroke="#8884d8" dot={false} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
};

export default WaveEquationSimulator;